import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.Chem.Draw import MolDraw2DSVG
from mordred import Calculator, descriptors
import pandas as pd
from autogluon.tabular import TabularPredictor
import tempfile
import base64
from io import BytesIO


# 添加 CSS 样式
st.markdown(
    """
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 50px auto;
        max-width: 39%; /* 设置最大宽度 */
        background-color: #f9f9f9f9;
        padding: 20px; /* 增加内边距 */
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
    a {
        color: #0000EE;
        text-decoration: underline;
    }
    .process-text, .molecular-weight {
        font-family: Arial, sans-serif;
        font-size: 16px;
        color: #333;
    }
    .stDataFrame {
        margin-top: 10px;
        margin-bottom: 0px !important;
    }
    .molecule-img {
        display: block;
        margin: 20px auto;
        max-width: 300px;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
        background-color: white;
    }
     /* 针对小屏幕的优化 */
    @media (max-width: 768px) {
        .rounded-container {
            padding: 10px; /* 减少内边距 */
        }
        .rounded-container blockquote {
            font-size: 0.9em; /* 缩小字体 */
        }
        .rounded-container h2 {
            font-size: 1.2em; /* 调整标题字体大小 */
        }
        .stApp {
            padding: 1px !important; /* 减少内边距 */
            max-width: 99%; /* 设置最大宽度 */
        }
        .process-text, .molecular-weight {
            font-size: 0.9em; /* 缩小文本字体 */
        }
        .molecule-img {
            max-width: 200px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 页面标题和简介
st.markdown(
    """
    <div class='rounded-container'>
        <h2>Predict Organic Fluorescence <br>Emission Wavelengths</h2>
        <blockquote>
            1. This website aims to quickly predict the emission wavelength of organic molecules based on their structure (SMILES) and solvent using machine learning models.<br>
            2. Code and data are available at <a href='https://github.com/dqzs/Fluorescence-Emission-Wavelength-Prediction' target='_blank'>https://github.com/dqzs/Fluorescence-Emission-Wavelength-Prediction</a>.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# 溶剂数据字典
solvent_data = {
    "H2O": {"Et30": 63.1, "SP": 0.681, "SdP": 0.997, "SA": 1.062, "SB": 0.025},
    "MeOH": {"Et30": 55.4, "SP": 0.608, "SdP": 0.904, "SA": 0.605, "SB": 0.545},
    "EtOH": {"Et30": 51.9, "SP": 0.633, "SdP": 0.783, "SA": 0.4, "SB": 0.658},
    "MeCN": {"Et30": 45.6, "SP": 0.645, "SdP": 0.974, "SA": 0.044, "SB": 0.286},
    "DMSO": {"Et30": 45.1, "SP": 0.83, "SdP": 1.0, "SA": 0.072, "SB": 0.647},
    "THF": {"Et30": 37.4, "SP": 0.714, "SdP": 0.634, "SA": 0.0, "SB": 0.591},
    "CH2Cl2": {"Et30": 40.7, "SP": 0.761, "SdP": 0.769, "SA": 0.04, "SB": 0.178},
    "CHCl3": {"Et30": 39.1, "SP": 0.783, "SdP": 0.614, "SA": 0.047, "SB": 0.071},
    "CCl4": {"Et30": 32.4, "SP": 0.768, "SdP": 0.0, "SA": 0.0, "SB": 0.044},
    "DMF": {"Et30": 43.2, "SP": 0.759, "SdP": 0.977, "SA": 0.031, "SB": 0.613},
    "DCE": {"Et30": 41.3, "SP": 0.771, "SdP": 0.742, "SA": 0.03, "SB": 0.126},
    "CS2": {"Et30": 32.8, "SP": 1.0, "SdP": 0.0, "SA": 0.0, "SB": 0.104},
    "1,4-dioxane": {"Et30": 36.0, "SP": 0.737, "SdP": 0.312, "SA": 0.0, "SB": 0.444},
    "Toluene": {"Et30": 33.9, "SP": 0.782, "SdP": 0.284, "SA": 0.0, "SB": 0.128},
    "Glycerol": {"Et30": 57.0, "SP": 0.828, "SdP": 0.921, "SA": 0.653, "SB": 0.309},
    "Hexane": {"Et30": 31.0, "SP": 0.616, "SdP": 0.0, "SA": 0.0, "SB": 0.056},
    "Acetic acid": {"Et30": 51.7, "SP": 0.651, "SdP": 0.676, "SA": 0.689, "SB": 0.39},
    "Acetone": {"Et30": 42.2, "SP": 0.651, "SdP": 0.907, "SA": 0.0, "SB": 0.475},
    "Ethyl acetate": {"Et30": 38.1, "SP": 0.656, "SdP": 0.603, "SA": 0.0, "SB": 0.542},
    "Benzene": {"Et30": 34.3, "SP": 0.793, "SdP": 0.27, "SA": 0.0, "SB": 0.124},
    "Cyclohexane": {"Et30": 30.9, "SP": 0.683, "SdP": 0.0, "SA": 0.0, "SB": 0.073},
    "Nitroethane": {"Et30": 43.6, "SP": 0.706, "SdP": 0.902, "SA": 0.0, "SB": 0.234},
    "Chlorobenzene": {"Et30": 36.8, "SP": 0.833, "SdP": 0.537, "SA": 0.0, "SB": 0.182},
    "Fluorobenzene": {"Et30": 37.0, "SP": 0.761, "SdP": 0.511, "SA": 0.0, "SB": 0.113},
    "Nitrobenzene": {"Et30": 41.2, "SP": 0.891, "SdP": 0.873, "SA": 0.056, "SB": 0.24},
    "Cyclohexanone": {"Et30": 39.8, "SP": 0.766, "SdP": 0.745, "SA": 0.0, "SB": 0.482},
    "Ethylene glycol": {"Et30": 56.3, "SP": 0.777, "SdP": 0.91, "SA": 0.717, "SB": 0.534},
    "Propionitrile": {"Et30": 43.6, "SP": 0.668, "SdP": 0.888, "SA": 0.03, "SB": 0.365},
    "Ethyl formate": {"Et30": 40.9, "SP": 0.648, "SdP": 0.707, "SA": 0.0, "SB": 0.477},
    "Methyl acetate": {"Et30": 38.9, "SP": 0.645, "SdP": 0.637, "SA": 0.0, "SB": 0.527},
    "Dimethyl carbonate": {"Et30": 38.2, "SP": 0.653, "SdP": 0.531, "SA": 0.064, "SB": 0.433},
    "1-propanol": {"Et30": 50.7, "SP": 0.658, "SdP": 0.748, "SA": 0.367, "SB": 0.782},
    "2-propanol": {"Et30": 48.4, "SP": 0.633, "SdP": 0.808, "SA": 0.283, "SB": 0.83},
    "1,2-propanediol": {"Et30": 54.1, "SP": 0.731, "SdP": 0.888, "SA": 0.475, "SB": 0.598},
    "Benzonitrile": {"Et30": 41.5, "SP": 0.851, "SdP": 0.852, "SA": 0.047, "SB": 0.281},
    "Heptane": {"Et30": 31.1, "SP": 0.635, "SdP": 0.0, "SA": 0.0, "SB": 0.083},
    "1-heptanol": {"Et30": 48.5, "SP": 0.706, "SdP": 0.499, "SA": 0.302, "SB": 0.912},
    "p-xylene": {"Et30": 33.1, "SP": 0.778, "SdP": 0.175, "SA": 0.0, "SB": 0.16},
    "Ethoxybenzene": {"Et30": 36.6, "SP": 0.81, "SdP": 0.669, "SA": 0.0, "SB": 0.295},
    "Octane": {"Et30": 31.1, "SP": 0.65, "SdP": 0.0, "SA": 0.0, "SB": 0.079},
    "Dibutyl ether": {"Et30": 33.0, "SP": 0.672, "SdP": 0.175, "SA": 0.0, "SB": 0.637},
    "Mesitylene": {"Et30": 32.9, "SP": 0.775, "SdP": 0.155, "SA": 0.0, "SB": 0.19},
    "Nonane": {"Et30": 31.0, "SP": 0.66, "SdP": 0.0, "SA": 0.0, "SB": 0.053},
    "1,2,3,4-tetrahydronaphthalene": {"Et30": 33.5, "SP": 0.838, "SdP": 0.182, "SA": 0.0, "SB": 0.18},
    "Decane": {"Et30": 31.0, "SP": 0.669, "SdP": 0.0, "SA": 0.0, "SB": 0.066},
    "1-decanol": {"Et30": 47.7, "SP": 0.722, "SdP": 0.383, "SA": 0.259, "SB": 0.912},
    "1-methylnaphthalene": {"Et30": 35.3, "SP": 0.908, "SdP": 0.51, "SA": 0.0, "SB": 0.156},
    "Dodecane": {"Et30": 31.1, "SP": 0.683, "SdP": 0.0, "SA": 0.0, "SB": 0.086},
    "Tributylamine": {"Et30": 32.1, "SP": 0.689, "SdP": 0.06, "SA": 0.0, "SB": 0.854},
    "Dibenzyl ether": {"Et30": 36.3, "SP": 0.877, "SdP": 0.509, "SA": 0.0, "SB": 0.33},
    "Trimethyl phosphate": {"Et30": 43.6, "SP": 0.707, "SdP": 0.909, "SA": 0.0, "SB": 0.522},
    "Butanenitrile": {"Et30": 42.5, "SP": 0.689, "SdP": 0.864, "SA": 0.0, "SB": 0.384},
    "2-butanone": {"Et30": 41.3, "SP": 0.669, "SdP": 0.872, "SA": 0.0, "SB": 0.52},
    "Sulfolane": {"Et30": 44.0, "SP": 0.83, "SdP": 0.896, "SA": 0.052, "SB": 0.365},
    "1-butanol": {"Et30": 49.7, "SP": 0.674, "SdP": 0.655, "SA": 0.341, "SB": 0.809},
    "Ethyl ether": {"Et30": 34.5, "SP": 0.617, "SdP": 0.385, "SA": 0.0, "SB": 0.562},
    "2-methyl-2-propanol": {"Et30": 43.3, "SP": 0.632, "SdP": 0.732, "SA": 0.145, "SB": 0.928},
    "1,2-dimethoxyethane": {"Et30": 38.2, "SP": 0.68, "SdP": 0.625, "SA": 0.0, "SB": 0.636},
    "Pyridine": {"Et30": 40.5, "SP": 0.842, "SdP": 0.761, "SA": 0.033, "SB": 0.581},
    "1-methyl-2-pyrrolidinone": {"Et30": 42.2, "SP": 0.812, "SdP": 0.959, "SA": 0.024, "SB": 0.613},
    "2-methyltetrahydrofuran": {"Et30": 36.5, "SP": 0.7, "SdP": 0.768, "SA": 0.0, "SB": 0.584},
    "2-pentanone": {"Et30": 41.1, "SP": 0.689, "SdP": 0.783, "SA": 0.01, "SB": 0.537},
    "3-pentanone": {"Et30": 39.3, "SP": 0.692, "SdP": 0.785, "SA": 0.0, "SB": 0.557},
    "Propyl acetate": {"Et30": 37.5, "SP": 0.67, "SdP": 0.559, "SA": 0.0, "SB": 0.548},
    "Piperidine": {"Et30": 35.5, "SP": 0.754, "SdP": 0.365, "SA": 0.0, "SB": 0.933},
    "2-methylbutane": {"Et30": 30.9, "SP": 0.581, "SdP": 0.0, "SA": 0.0, "SB": 0.053},
    "Pentane": {"Et30": 31.0, "SP": 0.593, "SdP": 0.0, "SA": 0.0, "SB": 0.073},
    "Tert-butyl methyl ether": {"Et30": 34.7, "SP": 0.622, "SdP": 0.422, "SA": 0.0, "SB": 0.567},
    "Hexafluorobenzene": {"Et30": 34.2, "SP": 0.623, "SdP": 0.252, "SA": 0.0, "SB": 0.119},
    "1,2-dichlorobenzene": {"Et30": 38.0, "SP": 0.869, "SdP": 0.676, "SA": 0.033, "SB": 0.144},
    "Bromobenzene": {"Et30": 36.6, "SP": 0.875, "SdP": 0.497, "SA": 0.0, "SB": 0.192},
    "N,N-dimethylacetamide": {"Et30": 42.9, "SP": 0.763, "SdP": 0.987, "SA": 0.028, "SB": 0.65},
    "Butyl acetate": {"Et30": 38.5, "SP": 0.674, "SdP": 0.535, "SA": 0.0, "SB": 0.525},
    "N,N-diethylacetamide": {"Et30": 41.4, "SP": 0.748, "SdP": 0.918, "SA": 0.0, "SB": 0.66},
    "Diisopropyl ether": {"Et30": 34.1, "SP": 0.625, "SdP": 0.324, "SA": 0.0, "SB": 0.657},
    "Dipropyl ether": {"Et30": 34.0, "SP": 0.645, "SdP": 0.286, "SA": 0.0, "SB": 0.666},
    "1-hexanol": {"Et30": 48.8, "SP": 0.698, "SdP": 0.552, "SA": 0.315, "SB": 0.879},
    "Hexamethylphosphoramine": {"Et30": 40.9, "SP": 0.744, "SdP": 1.1, "SA": 0.0, "SB": 0.813},
    "Triethylamine": {"Et30": 32.1, "SP": 0.66, "SdP": 0.108, "SA": 0.0, "SB": 0.885},
    "(Trifluoromethyl)benzene": {"Et30": 38.5, "SP": 0.694, "SdP": 0.663, "SA": 0.014, "SB": 0.073},
    "1,1,2-trichlorotrifluoroethane": {"Et30": 33.2, "SP": 0.596, "SdP": 0.152, "SA": 0.0, "SB": 0.038},
    "1,1,2,2-tetrachloroethane": {"Et30": 39.4, "SP": 0.845, "SdP": 0.792, "SA": 0.0, "SB": 0.017},
    "1,1,1-trichloroethane": {"Et30": 36.2, "SP": 0.737, "SdP": 0.5, "SA": 0.0, "SB": 0.085},
    "2,2,2-trifluoroethanol": {"Et30": 59.8, "SP": 0.543, "SdP": 0.922, "SA": 0.893, "SB": 0.107},
    "dioxane": {"Et30": 36.0, "SP": 0.737, "SdP": 0.312, "SA": 0.0, "SB": 0.444},
    "Me-THF": {"Et30": 36.5, "SP": 0.7, "SdP": 0.768, "SA": 0.0, "SB": 0.584},
    "DCB": {"Et30": 38.0, "SP": 0.869, "SdP": 0.676, "SA": 0.033, "SB": 0.144},
    "DMA": {"Et30": 42.9, "SP": 0.763, "SdP": 0.987, "SA": 0.028, "SB": 0.65},
    "N-Methylformamide": {"Et30": 54.1, "SP": 0.759, "SdP": 0.977, "SA": 0.031, "SB": 0.613}
}

# 溶剂选择下拉菜单
solvent = st.selectbox("Select Solvent:", list(solvent_data.keys()))

# SMILES 输入区域
smiles = st.text_input("Enter the SMILES representation of the molecule:", placeholder="e.g., NC1=CC=C(C=C1)C(=O)O")

# 提交按钮
submit_button = st.button("Submit and Predict", key="predict_button")

# 指定的描述符列表
required_descriptors = ["MAXdssC", "VSA_EState7", "SMR_VSA10", "PEOE_VSA8"]

# 分子图像路径
image_path = None


def mol_to_image(mol, size=(300, 300)):
    """将分子转换为图像并返回base64编码"""
    d2d = Draw.MolDraw2DSVG(size[0], size[1])
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    return svg


def get_descriptors(mol):
    """获取指定的分子描述符"""
    # 计算RDKit描述符
    try:
        rdkit_descs = {
            "VSA_EState7": Descriptors.VSA_EState7(mol),
            "SMR_VSA10": Descriptors.SMR_VSA10(mol),
            "PEOE_VSA8": Descriptors.PEOE_VSA8(mol),
        }
    except:
        # 如果计算失败，使用默认值
        rdkit_descs = {
            "VSA_EState7": 0.0,
            "SMR_VSA10": 0.0,
            "PEOE_VSA8": 0.0,
        }

    # 计算Mordred描述符
    try:
        calc = Calculator(descriptors, ignore_3D=True)
        mordred_desc = calc(mol)
        maxdssc = mordred_desc["MAXdssC"] if "MAXdssC" in mordred_desc else 0.0
    except:
        maxdssc = 0.0

    return {
        "MAXdssC": maxdssc,
        **rdkit_descs
    }


# 如果点击提交按钮
if submit_button:
    if not smiles:
        st.error("Please enter a valid SMILES string.")
    elif not solvent:
        st.error("Please select a solvent.")
    else:
        with st.spinner("Processing molecule and making predictions..."):
            try:
                # 处理SMILES输入
                st.markdown('<div class="process-text">Processing SMILES input...</div>', unsafe_allow_html=True)
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # 添加H原子并生成2D坐标
                    mol = Chem.AddHs(mol)
                    AllChem.Compute2DCoords(mol)

                    # 显示分子结构
                    svg = mol_to_image(mol)
                    st.markdown(f'<div class="molecule-img">{svg}</div>', unsafe_allow_html=True)

                    # 计算分子量
                    mol_weight = Descriptors.MolWt(mol)
                    st.markdown(f'<div class="molecular-weight">Molecular Weight: {mol_weight:.2f} g/mol</div>',
                                unsafe_allow_html=True)

                    # 获取溶剂参数
                    solvent_params = solvent_data[solvent]

                    # 计算指定描述符
                    st.info("Calculating molecular descriptors...")
                    desc_values = get_descriptors(mol)

                    # 创建输入数据表
                    input_data = {
                        "SMILES": [smiles],
                        "Et30": [solvent_params["Et30"]],
                        "SP": [solvent_params["SP"]],
                        "SdP": [solvent_params["SdP"]],
                        "SA": [solvent_params["SA"]],
                        "SB": [solvent_params["SB"]],
                        "MAXdssC": [desc_values["MAXdssC"]],
                        "VSA_EState7": [desc_values["VSA_EState7"]],
                        "SMR_VSA10": [desc_values["SMR_VSA10"]],
                        "PEOE_VSA8": [desc_values["PEOE_VSA8"]],
                        "image": ["Molecular Structure"]
                    }

                    input_df = pd.DataFrame(input_data)

                    # 显示输入数据
                    st.write("Input Data:")
                    st.dataframe(input_df)

                    # 创建预测用数据框
                    predict_data = {
                        "Et30": [solvent_params["Et30"]],
                        "SP": [solvent_params["SP"]],
                        "SdP": [solvent_params["SdP"]],
                        "SA": [solvent_params["SA"]],
                        "SB": [solvent_params["SB"]],
                        "MAXdssC": [desc_values["MAXdssC"]],
                        "VSA_EState7": [desc_values["VSA_EState7"]],
                        "SMR_VSA10": [desc_values["SMR_VSA10"]],
                        "PEOE_VSA8": [desc_values["PEOE_VSA8"]]
                    }

                    predict_df = pd.DataFrame(predict_data)

                    # 加载模型并预测
                    st.info("Loading the model and predicting the emission wavelength...")
                    predictor = TabularPredictor.load("./ag-20250529_123557")

                    # 指定模型列表
                    model_options = ['LightGBM',
                                     'LightGBMXT',
                                     'CatBoost',
                                     'XGBoost',
                                     'NeuralNetTorch',
                                     'LightGBMLarge',
                                     'MultiModalPredictor',
                                     'WeightedEnsemble_L2'
                                    ]
                    

                    # 获取预测结果
                    predictions_dict = {}
                    for model in model_options:
                        predictions = predictor.predict(input_df, model=model)
                        predictions_dict[model] = predictions.astype(int).apply(lambda x: f"{x} nm")
                        st.error(f"1")

                    # 显示预测结果
                    st.write("Prediction Results:")
                    st.markdown(
                        "**Note:** WeightedEnsemble_L2 is a meta-model combining predictions from other models.")
                    results_df = pd.DataFrame(predictions_dict)
                    st.dataframe(results_df)

                else:
                    st.error("Invalid SMILES input. Please check the format.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")