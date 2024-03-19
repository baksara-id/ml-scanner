from tensorflow import keras

# MODEL_PATH = './save_model/'
MODEL_PATH = './api_model/save_model/'
MODEL = keras.models.load_model(MODEL_PATH+'model.h5')

CLASS_NAMES = ['carakan_ba', 'carakan_ca', 'carakan_da', 'carakan_dha', 'carakan_ga', 'carakan_ha',
                    'carakan_ja', 'carakan_ka', 'carakan_la', 'carakan_ma', 'carakan_na', 'carakan_nga',
                    'carakan_nya', 'carakan_pa', 'carakan_ra', 'carakan_sa', 'carakan_ta', 'carakan_tha',
                    'carakan_wa', 'carakan_ya', 'sandhangan_e', 'sandhangan_e2', 'sandhangan_h', 'sandhangan_i',
                    'sandhangan_ng', 'sandhangan_o', 'sandhangan_r', 'sandhangan_u']

# Path: api_model/BaksaraConst.py

MODELS = [
    keras.models.load_model(MODEL_PATH+'model_1.h5'),
    keras.models.load_model(MODEL_PATH+'model_2.h5'),
    keras.models.load_model(MODEL_PATH+'model_3.h5'),
    keras.models.load_model(MODEL_PATH+'model_4.h5'),
]
# first class is ha na ca ra ka in sorted
# second class is da la sa ta wa
# third class is dha ja nya pa ya
# fourth class is ba ga ma nga tha
CLASS_NAMES4 = [
    ['carakan_ca', 'carakan_ha', 'carakan_ka', 'carakan_na', 'carakan_ra'],
    ['carakan_da', 'carakan_la', 'carakan_sa', 'carakan_ta', 'carakan_wa'],
    ['carakan_dha', 'carakan_ja', 'carakan_nya', 'carakan_pa', 'carakan_ya'],
    ['carakan_ba', 'carakan_ga', 'carakan_ma', 'carakan_nga', 'carakan_tha'],
]

TheBypass = ['carakan_na', 'carakan_sa','carakan_wa','carakan_ba','carakan_ca','carakan_dha','carakan_tha']