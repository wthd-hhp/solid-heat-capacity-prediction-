import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Draw import MolDraw2DSVG
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
import pandas as pd
from autogluon.tabular import TabularPredictor
import numpy as np
import gc
import re
import traceback

# ---------------- 页面样式 ----------------
st.markdown(
    """
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 50px auto;
        max-width: 40%;
        background-color: #f9f9f9f9;
        padding: 20px;
        box-sizing: border-box;
    }
    .rounded-container h2 {
        margin-top: -80px;
        text-align: center;
        background-color: #e0e0e0e0;
        padding: 10px;
        border-radius: 10px;
    }
    .rounded-container blockquote {
        text-align: left;
        margin: 20px auto;
        background-color: #f0f0f0;
        padding: 10px;
        font-size: 1.1em;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- 页面标题 ----------------
st.markdown(
    """
    <div class='rounded-container'>
        <h2>Predict Heat Capacity (Cp) of Organic Molecules</h2>
        <blockquote>
            1. This web app predicts the heat capacity (Cp) of organic molecules based on their SMILES structure using trained machine learning models.<br>
            2. Enter a valid SMILES string below to get the predicted result.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- 模型配置 ----------------
MODEL_CONFIG = {
    "Gas": {
        "path": "./autogluon/gas",
        "features": ["ATS0se", "EState_VSA5", "ATSC0dv"]
    },
    "Liquid": {
        "path": "./autogluon/liquid",
        "features": ["ATS0s", "PEOE_VSA6", "SssCH2"]
    },
    "Solid": {
        "path": "./autogluon/solid",
        "features": ["ATSC0dv", "ATS0s", "ATS0pe"]
    }
}


ESSENTIAL_MODELS = [
    "CatBoost_BAG_L1",
    "LightGBM_BAG_L1",
    "LightGBMLarge_BAG_L1",
    "MultiModalPredictor_BAG_L1",
    "XGBoost_BAG_L1",
]


# ---------------- 选择物态 ----------------
state = st.selectbox(
    "Select the physical state of the substance:",
    ("Gas", "Liquid", "Solid"),
)

# ---------------- 用户输入 ----------------
smiles = st.text_input(
    "Enter the SMILES representation of the molecule:",
    placeholder="e.g., C1=CC=CC=C1O",
)

submit_button = st.button("Submit and Predict")




# ---------------- 模型加载 ----------------
@st.cache_resource(show_spinner=False)
def load_predictor(model_path):
    """根据物态加载 AutoGluon 模型"""
    return TabularPredictor.load(model_path)

# ---------------- 分子绘图 ----------------
def mol_to_image(mol, size=(300, 300)):
    d2d = MolDraw2DSVG(size[0], size[1])
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    svg = re.sub(r"<rect[^>]*>", "", svg, flags=re.DOTALL)
    return svg



# ---------------- RDKit 描述符 ----------------
def calc_rdkit_descriptors(smiles_list):
    desc_names = [name for name, _ in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        desc = calc.CalcDescriptors(mol)
        results.append(desc)
    return pd.DataFrame(results, columns=desc_names)

# ---------------- Mordred 描述符 ----------------
def calc_mordred_descriptors(smiles_list):
    calc = Calculator(descriptors, ignore_3D=True)
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        res = calc(mol)
        desc_dict = {}
        for key, val in res.asdict().items():
            if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
                desc_dict[key] = val[0] if len(val) > 0 else np.nan
            elif val is None or isinstance(val, complex):
                desc_dict[key] = np.nan
            elif hasattr(val, "__class__") and val.__class__.__name__ == "Missing":
                desc_dict[key] = np.nan
            else:
                desc_dict[key] = val
        results.append(desc_dict)
    return pd.DataFrame(results)


# ---------------- 特征合并 ----------------
def merge_features_without_duplicates(original_df, *feature_dfs):
    merged = pd.concat([original_df] + list(feature_dfs), axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    # 新增：把 list/ndarray 压成标量
    merged = merged.applymap(lambda x: float(np.mean(x)) if isinstance(x, (list, np.ndarray, tuple)) else float(x))
    return merged





# ---------------- 主预测逻辑 ----------------
if submit_button:
    if not smiles:
        st.error("Please enter a valid SMILES string.")
    else:
        with st.spinner("Processing molecule and predicting heat capacity..."):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    st.error("Invalid SMILES format.")
                    st.stop()

                # 显示分子结构
                mol = Chem.AddHs(mol)
                AllChem.Compute2DCoords(mol)
                svg = mol_to_image(mol)
                st.markdown(f'<div style="text-align:center;">{svg}</div>', unsafe_allow_html=True)

                # 分子量
                mol_weight = Descriptors.MolWt(mol)
                st.markdown(f"**Molecular Weight:** {mol_weight:.2f} g/mol")

                # 计算描述符
                smiles_list = [smiles]
                rdkit_features = calc_rdkit_descriptors(smiles_list)
                mordred_features = calc_mordred_descriptors(smiles_list)
                merged_features = merge_features_without_duplicates(rdkit_features, mordred_features)

                # 获取当前状态的模型与特征
                config = MODEL_CONFIG[state]
                required_features = config["features"]
                model_path = config["path"]
               # 自动检测缺失列
                missing_cols = [f for f in required_features if f not in merged_features.columns]
                if missing_cols:
                    st.error(f"Missing descriptors: {missing_cols}")
                    st.stop()

                 # --- 创建输入数据表（含 SMILES）---
                input_data = {"SMILES": [smiles]}
                for f in feature_names:
                    input_data[f] = [merged_features.iloc[0][f]]
                input_df = pd.DataFrame(input_data)
                
                

                # 显示输入特征表
                st.write(f"Input Features for {state} model:")
                st.dataframe(input_df)

                # --- 仅取特征列进行预测 ---
                predict_df = merged_features.loc[:, feature_names]

                # 加载对应模型并预测
                model_path = MODEL_PATHS[state]
                predictor = load_predictor(model_path)
                

                 # --- 多模型预测 ---
                predictions_dict = {}
                for model in ESSENTIAL_MODELS:
                    try:
                        pred = predictor.predict(predict_df, model=model)
                        predictions_dict[model] = pred.astype(float).apply(lambda x: f"{x:.2f} J/(mol·K)")
                    except Exception as model_error:
                        st.warning(f"Model {model} prediction failed: {str(model_error)}")
                        predictions_dict[model] = "Error"

                # --- 展示结果 ---
                st.write(f"Prediction Results ({state} Models):")
                results_df = pd.DataFrame(predictions_dict)
                st.dataframe(results_df)

                # 主动释放内存
                del predictor
                gc.collect()

            except Exception as e:
                st.error(f"Model loading failed: {str(e)}")

 

















