import numpy as np
import pandas as pd
# import pyspark
# import joblib
import pickle
import random
#
import time
# import lightgbm as lgb
# import shap
# import optuna
# from optuna.samplers import TPESampler

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

random.seed(10)

def write_to_pickle(data, output_filename):
    output = Transforms.get_output()
    output_fs = output.filesystem()

    with output_fs.open(output_filename + '.pickle', 'wb') as f:
        pickle.dump(data, f)



def read_from_pickle(transform_input, filename):
    with transform_input.filesystem().open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def n3c_score(gbm, X, y, dic_partner_pidx_valid):
    # iterate through validation sites for AUROC term
    lst_auroc = []
    for partner in dic_partner_pidx_valid:
        X_temp = X[dic_partner_pidx_valid[partner]]
        y_temp = y[dic_partner_pidx_valid[partner]]
        preds = gbm.predict(X_temp)
        try:
            lst_auroc.append(roc_auc_score(y_temp, preds))
        except:
            continue
    mean_auroc = np.mean(lst_auroc)
    var_auroc = np.var(lst_auroc)
    term_auroc = mean_auroc - var_auroc
    # AUPR and F2
    preds = gbm.predict(X)
    term_aupr = average_precision_score(y, preds)
    precision, recall, thresholds = precision_recall_curve(y, preds)
    precision = precision + 0.0001
    recall = recall + 0.0001
    f2 = 5*np.multiply(precision, recall)/(4*precision+recall)
    term_f2 = np.max(f2)
    # customized score defined in instructions
    # score = round(term_auroc,5) + round(term_f2,5) + round(term_aupr,5)
    # return evaluation
    return round(term_auroc,5) + round(term_f2,5) + round(term_aupr,5)

def icd10block_to_chapter(icd10_block):
    if icd10_block >= 1 and icd10_block <= 22:
        res = 1
    if (icd10_block >= 23 and icd10_block <= 42) or icd10_block in [285, 287]:
        res = 2
    if icd10_block >= 43 and icd10_block <= 49:
        res = 3
    if icd10_block >= 50 and icd10_block <= 59:
        res = 4
    if icd10_block >= 60 and icd10_block <= 70:
        res = 5
    if icd10_block >= 71 and icd10_block <= 81:
        res = 6
    if icd10_block >= 82 and icd10_block <= 93:
        res = 7
    if icd10_block >= 94 and icd10_block <= 98:
        res = 8
    if icd10_block >= 99 and icd10_block <= 108:
        res = 9
    if icd10_block >= 109 and icd10_block <= 119:
        res = 10
    if icd10_block >= 120 and icd10_block <= 129:
        res = 11
    if icd10_block >= 130 and icd10_block <= 138:
        res = 12
    if icd10_block >= 139 and icd10_block <= 158:
        res = 13
    if icd10_block >= 159 and icd10_block <= 169:
        res = 14
    if icd10_block >= 170 and icd10_block <= 178:
        res = 15
    if icd10_block >= 179 and icd10_block <= 190:
        res = 16
    if icd10_block >= 191 and icd10_block <= 201:
        res = 17
    if icd10_block >= 202 and icd10_block <= 215:
        res = 18
    if icd10_block >= 216 and icd10_block <= 225:
        res = 19
    if (icd10_block >= 226 and icd10_block <= 237) or icd10_block == 286:
        res = 20
    if icd10_block >= 238 and icd10_block <= 239:
        res = 21
    if icd10_block >= 240 and icd10_block <= 251:
        res = 22
    if icd10_block >= 252 and icd10_block <= 256:
        res = 23
    if icd10_block >= 257 and icd10_block <= 263:
        res = 24
    if icd10_block >= 264 and icd10_block <= 269:
        res = 25
    if icd10_block >= 270 and icd10_block <= 284:
        res = 26
    return res

def icd10_encode(icd10):
	
    icd10 = icd10[:3]

    if icd10 == 'C7A':
        return 35
    if icd10 == 'C7B':
        return 36
    if icd10 == 'D3A':
        return 41
    if icd10 == 'I5A': # included in I30-I5A
        return 104
    if icd10 == 'O9A': # included in O94-O9A
        return 178
    if icd10 == 'Z3A':
        return 277

    c = icd10[0].upper()
    n = int(icd10[1:])
    # Certain infectious and parasitic diseases
    if c == 'A':
        if n >= 0 and n <= 9:
            res = 1
        if n >= 15 and n <= 19:
            res = 2
        if n >= 20 and n <= 28:
            res = 3
        if n >= 30 and n <= 49:
            res = 4
        if n >= 50 and n <= 64:
            res = 5
        if n >= 65 and n <= 69:
            res = 6
        if n >= 70 and n <= 74:
            res = 7
        if n >= 75 and n <= 79:
            res = 8
        if n >= 80 and n <= 89:
            res = 9
        if n >= 90 and n <= 99:
            res = 10
    if c == 'B':
        if n >= 0 and n <= 9:
            res = 11
        if n == 10:
            res = 12
        if n >= 15 and n <= 19:
            res = 13
        if n >= 20 and n <= 24:
            res = 14
        if n >= 25 and n <= 34:
            res = 15
        if n >= 35 and n <= 49:
            res = 16
        if n >= 50 and n <= 64:
            res = 17
        if n >= 65 and n <= 84:
            res = 18
        if n >= 85 and n <= 89:
            res = 19
        if n >= 90 and n <= 94:
            res = 20
        if n >= 95 and n <= 97:
            res = 21
        if n >= 98 and n <= 99:
            res = 22
    # Neoplasms
    if c == 'C':
        if n >= 0 and n <= 14:
            res = 23
        if n >= 15 and n <= 26:
            res = 24
        if n >= 30 and n <= 39:
            res = 25
        if n >= 40 and n <= 41:
            res = 26
        if n >= 43 and n <= 44:
            res = 27
        if n >= 45 and n <= 49:
            res = 28
        if n == 50:
            res = 29
        if n >= 51 and n <= 58:
            res = 30
        if n >= 60 and n <= 63:
            res = 31
        if n >= 64 and n <= 68:
            res = 32
        if n >= 69 and n <= 72:
            res = 33
        if n >= 73 and n <= 75:
            res = 34
        if n >= 76 and n <= 80:
            res = 285
        if n >= 81 and n <= 96:
            res = 37
        if n == 97:
            res = 287
    if c == 'D':
        if n >= 0 and n <= 9:
            res = 38
        if n >= 10 and n <= 36:
            res = 39
        if n >= 37 and n <= 48:
            res = 40
        if n == 49:
            res = 42
    # Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism
        if n >= 50 and n <= 53:
            res = 43
        if n >= 55 and n <= 59:
            res = 44
        if n >= 60 and n <= 64:
            res = 45
        if n >= 65 and n <= 69:
            res = 46
        if n >= 70 and n <= 77:
            res = 47
        if n == 78:
            res = 48
        if n >= 80 and n <= 89:
            res = 49
    # Endocrine, nutritional and metabolic diseases
    if c == 'E':
        if n >= 0 and n <= 7:
            res = 50
        if n >= 8 and n <= 14:
            res = 51
        if n >= 15 and n <= 16:
            res = 52
        if n >= 20 and n <= 35:
            res = 53
        if n == 36:
            res = 54
        if n >= 40 and n <= 46:
            res = 55
        if n >= 50 and n <= 64:
            res = 56
        if n >= 65 and n <= 68:
            res = 57
        if n >= 70 and n <= 88:
            res = 58
        if n == 89:
            res = 59
        if n == 90:
            res = 58 # edited
    # Mental, Behavioral and Neurodevelopmental disorders
    if c == 'F':
        if n >= 0 and n <= 9:
            res = 60
        if n >= 10 and n <= 19:
            res = 61
        if n >= 20 and n <= 29:
            res = 62
        if n >= 30 and n <= 39:
            res = 63
        if n >= 40 and n <= 48:
            res = 64
        if n >= 50 and n <= 59:
            res = 65
        if n >= 60 and n <= 69:
            res = 66
        if n >= 70 and n <= 79:
            res = 67
        if n >= 80 and n <= 89:
            res = 68
        if n >= 90 and n <= 98:
            res = 69
        if n == 99:
            res = 70
    # Diseases of the nervous system
    if c == 'G':
        if n >= 0 and n <= 9:
            res = 71
        if n >= 10 and n <= 14:
            res = 72
        if n >= 20 and n <= 26:
            res = 73
        if n >= 30 and n <= 32:
            res = 74
        if n >= 35 and n <= 37:
            res = 75
        if n >= 40 and n <= 47:
            res = 76
        if n >= 50 and n <= 59:
            res = 77
        if n >= 60 and n <= 65:
            res = 78
        if n >= 70 and n <= 73:
            res = 79
        if n >= 80 and n <= 83:
            res = 80
        if n >= 89 and n <= 99:
            res = 81
    # Diseases of the eye and adnexa
    if c == 'H':
        if n >= 0 and n <= 6:
            res = 82
        if n >= 10 and n <= 13:
            res = 83
        if n >= 15 and n <= 22:
            res = 84
        if n >= 25 and n <= 28:
            res = 85
        if n >= 30 and n <= 36:
            res = 86
        if n >= 40 and n <= 42:
            res = 87
        if n >= 43 and n <= 45:
            res = 88
        if n >= 46 and n <= 48:
            res = 89
        if n >= 49 and n <= 52:
            res = 90
        if n >= 53 and n <= 54:
            res = 91
        if n >= 55 and n <= 57:
            res = 92
        if n == 59:
            res = 93
    # Diseases of the ear and mastoid process
        if n >= 60 and n <= 62:
            res = 94
        if n >= 65 and n <= 75:
            res = 95
        if n >= 80 and n <= 83:
            res = 96
        if n >= 90 and n <= 94:
            res = 97
        if n == 95:
            res = 98
    # Diseases of the circulatory system
    if c == 'I':
        if n >= 0 and n <= 2:
            res = 99
        if n >= 5 and n <= 9:
            res = 100
        if n >= 10 and n <= 16:
            res = 101
        if n >= 20 and n <= 25:
            res = 102
        if n >= 26 and n <= 28:
            res = 103
        if n >= 30 and n <= 52:
            res = 104
        if n >= 60 and n <= 69:
            res = 105
        if n >= 70 and n <= 79:
            res = 106
        if n >= 80 and n <= 89:
            res = 107
        if n >= 95 and n <= 99:
            res = 108
    # Diseases of the respiratory system
    if c == 'J':
        if n >= 0 and n <= 6:
            res = 109
        if n >= 9 and n <= 18:
            res = 110
        if n >= 20 and n <= 22:
            res = 111
        if n >= 30 and n <= 39:
            res = 112
        if n >= 40 and n <= 47:
            res = 113
        if n >= 60 and n <= 70:
            res = 114
        if n >= 80 and n <= 84:
            res = 115
        if n >= 85 and n <= 86:
            res = 116
        if n >= 90 and n <= 94:
            res = 117
        if n == 95:
            res = 118
        if n >= 96 and n <= 99:
            res = 119
    # Diseases of the digestive system
    if c == 'K':
        if n >= 0 and n <= 14:
            res = 120
        if n >= 20 and n <= 31:
            res = 121
        if n >= 35 and n <= 38:
            res = 122
        if n >= 40 and n <= 46:
            res = 123
        if n >= 50 and n <= 52:
            res = 124
        if n >= 55 and n <= 64:
            res = 125
        if n >= 65 and n <= 68:
            res = 126
        if n >= 70 and n <= 77:
            res = 127
        if n >= 80 and n <= 87:
            res = 128
        if n >= 90 and n <= 95:
            res = 129
    # Diseases of the skin and subcutaneous tissue
    if c == 'L':
        if n >= 0 and n <= 8:
            res = 130
        if n >= 10 and n <= 14:
            res = 131
        if n >= 20 and n <= 30:
            res = 132
        if n >= 40 and n <= 45:
            res = 133
        if n >= 49 and n <= 54:
            res = 134
        if n >= 55 and n <= 59:
            res = 135
        if n >= 60 and n <= 75:
            res = 136
        if n == 76:
            res = 137
        if n >= 80 and n <= 99:
            res = 138
    # Diseases of the musculoskeletal system and connective tissue
    if c == 'M':
        if n >= 0 and n <= 3:
            res = 139
        if n == 4:
            res = 140
        if n >= 5 and n <= 14:
            res = 141
        if n >= 15 and n <= 19:
            res = 142
        if n >= 20 and n <= 25:
            res = 143
        if n >= 26 and n <= 27:
            res = 144
        if n >= 30 and n <= 36:
            res = 145
        if n >= 40 and n <= 43:
            res = 146
        if n >= 45 and n <= 49:
            res = 147
        if n >= 50 and n <= 54:
            res = 148
        if n >= 60 and n <= 64:
            res = 149
        if n >= 65 and n <= 69:
            res = 150
        if n >= 70 and n <= 79:
            res = 151
        if n >= 80 and n <= 85:
            res = 152
        if n >= 86 and n <= 90:
            res = 153
        if n >= 91 and n <= 94:
            res = 154
        if n == 95:
            res = 155
        if n == 96:
            res = 156
        if n == 97:
            res = 157
        if n == 99:
            res = 158
    # Diseases of the genitourinary system
    if c == 'N':
        if n >= 0 and n <= 8:
            res = 159
        if n >= 10 and n <= 16:
            res = 160
        if n >= 17 and n <= 19:
            res = 161
        if n >= 20 and n <= 23:
            res = 162
        if n >= 25 and n <= 29:
            res = 163
        if n >= 30 and n <= 39:
            res = 164
        if n >= 40 and n <= 53:
            res = 165
        if n >= 60 and n <= 65:
            res = 166
        if n >= 70 and n <= 77:
            res = 167
        if n >= 80 and n <= 98:
            res = 168
        if n == 99:
            res = 169
    # Pregnancy, childbirth and the puerperium
    if c == 'O':
        if n >= 0 and n <= 8:
            res = 170
        if n == 9:
            res = 171
        if n >= 10 and n <= 16:
            res = 172
        if n >= 20 and n <= 29:
            res = 173
        if n >= 30 and n <= 48:
            res = 174
        if n >= 60 and n <= 77:
            res = 175
        if n >= 80 and n <= 84:
            res = 176
        if n >= 85 and n <= 92:
            res = 177
        if n >= 94 and n <= 99:
            res = 178
    # Certain conditions originating in the perinatal period
    if c == 'P':
        if n >= 0 and n <= 4:
            res = 179
        if n >= 5 and n <= 8:
            res = 180
        if n == 9:
            res = 181
        if n >= 10 and n <= 15:
            res = 182
        if n >= 19 and n <= 29:
            res = 183
        if n >= 35 and n <= 39:
            res = 184
        if n >= 50 and n <= 61:
            res = 185
        if n >= 70 and n <= 74:
            res = 186
        if n >= 75 and n <= 78:
            res = 187
        if n >= 80 and n <= 83:
            res = 188
        if n == 84:
            res = 189
        if n >= 90 and n <= 96:
            res = 190
    # Congenital malformations, deformations and chromosomal abnormalities
    if c == 'Q':
        if n >= 0 and n <= 7:
            res = 191
        if n >= 10 and n <= 18:
            res = 192
        if n >= 20 and n <= 28:
            res = 193
        if n >= 30 and n <= 34:
            res = 194
        if n >= 35 and n <= 37:
            res = 195
        if n >= 38 and n <= 45:
            res = 196
        if n >= 50 and n <= 56:
            res = 197
        if n >= 60 and n <= 64:
            res = 198
        if n >= 65 and n <= 79:
            res = 199
        if n >= 80 and n <= 89:
            res = 200
        if n >= 90 and n <= 99:
            res = 201
    # Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified
    if c == 'R':
        if n >= 0 and n <= 9:
            res = 202
        if n >= 10 and n <= 19:
            res = 203
        if n >= 20 and n <= 23:
            res = 204
        if n >= 25 and n <= 29:
            res = 205
        if n >= 30 and n <= 39:
            res = 206
        if n >= 40 and n <= 46:
            res = 207
        if n >= 47 and n <= 49:
            res = 208
        if n >= 50 and n <= 69:
            res = 209
        if n >= 70 and n <= 79:
            res = 210
        if n >= 80 and n <= 82:
            res = 211
        if n >= 83 and n <= 89:
            res = 212
        if n >= 90 and n <= 94:
            res = 213
        if n == 97:
            res = 214
        if n == 99 or n == 95 or n == 96 or n == 98:
            res = 215
    # Injury, poisoning and certain other consequences of external causes
    if c == 'S':
        if n >= 0 and n <= 9:
            res = 216
        if n >= 10 and n <= 19:
            res = 217
        if n >= 20 and n <= 29:
            res = 218
        if n >= 30 and n <= 39:
            res = 219
        if n >= 40 and n <= 49:
            res = 220
        if n >= 50 and n <= 59:
            res = 221
        if n >= 60 and n <= 69:
            res = 222
        if n >= 70 and n <= 79:
            res = 223
        if n >= 80 and n <= 89:
            res = 224
        if n >= 90 and n <= 99:
            res = 225
    if c == 'T':
        if n >= 0 and n <= 7:
            res = 226
        if n >= 8 and n <= 14:
            res = 227
        if n >= 15 and n <= 19:
            res = 228
        if n >= 20 and n <= 25:
            res = 229
        if n >= 26 and n <= 28:
            res = 230
        if n >= 29 and n <= 32:
            res = 231
        if n >= 33 and n <= 35:
            res = 232
        if n >= 36 and n <= 50:
            res = 233
        if n >= 51 and n <= 65:
            res = 234
        if n >= 66 and n <= 78:
            res = 235
        if n == 79:
            res = 236
        if n >= 80 and n <= 88:
            res = 237
        if n >= 90 and n <= 98:
            res = 286
    # Codes for special purposes
    if c == 'U':
        if n >= 0 and n <= 49:
            res = 238
        if n >= 50 and n <= 89:
            res = 239
    # External causes of morbidity
    if c == 'V':
        if n >= 0 and n <= 9:
            res = 240
        if n >= 10 and n <= 19:
            res = 241
        if n >= 20 and n <= 29:
            res = 242
        if n >= 30 and n <= 39:
            res = 243
        if n >= 40 and n <= 49:
            res = 244
        if n >= 50 and n <= 59:
            res = 245
        if n >= 60 and n <= 69:
            res = 246
        if n >= 70 and n <= 79:
            res = 247
        if n >= 80 and n <= 89:
            res = 248
        if n >= 90 and n <= 94:
            res = 249
        if n >= 95 and n <= 97:
            res = 250
        if n >= 98 and n <= 99:
            res = 251
    if c == 'W':
        if n >= 0 and n <= 19:
            res = 252
        if n >= 20 and n <= 49:
            res = 253
        if n >= 50 and n <= 64:
            res = 254
        if n >= 65 and n <= 84:
            res = 255
        if n >= 85 and n <= 99:
            res = 256
    if c == 'X':
        if n >= 0 and n <= 9:
            res = 257
        if n >= 10 and n <= 29:
            res = 258
        if n >= 30 and n <= 39:
            res = 259
        if n >= 40 and n <= 49:
            res = 260
        if n >= 50 and n <= 59:
            res = 261
        if n >= 60 and n <= 91:
            res = 262
        if n >= 92 and n <= 99:
            res = 263
    if c == 'Y':
        if n >= 0 and n <= 9:
            res = 263
        if n >= 10 and n <= 34:
            res = 264
        if n >= 35 and n <= 38:
            res = 265
        if n >= 40 and n <= 69:
            res = 266
        if n >= 70 and n <= 82:
            res = 267
        if n >= 83 and n <= 84:
            res = 268
        if n >= 85 and n <= 99:
            res = 269
    # Factors influencing health status and contact with health services
    if c == 'Z':
        if n >= 0 and n <= 13:
            res = 270
        if n >= 14 and n <= 15:
            res = 271
        if n == 16:
            res = 272
        if n == 17:
            res = 273
        if n == 18:
            res = 274
        if n == 19:
            res = 275
        if n >= 20 and n <= 29:
            res = 276
        if n >= 30 and n <= 39:
            res = 277
        if n >= 40 and n <= 54:
            res = 278
        if n >= 55 and n <= 65:
            res = 279
        if n == 66:
            res = 280
        if n == 67:
            res = 281
        if n == 68:
            res = 282
        if n >= 69 and n <= 76:
            res = 283
        if n >= 77 and n <= 99:
            res = 284

    return res

def atc_encode(atc):
    atc = atc[:3]
    c = atc[0].upper()
    n = int(atc[1:])

    if c == 'A': # Alimentary tract and metabolisma
        if n == 1:
            res = 1
        if n == 2:
            res = 2
        if n == 3:
            res = 3
        if n == 4:
            res = 4
        if n == 5:
            res = 5
        if n == 6:
            res = 6
        if n == 7:
            res = 7
        if n == 8:
            res = 8
        if n == 9:
            res = 9
        if n == 10:
            res = 10
        if n == 11:
            res = 11
        if n == 12:
            res = 12
        if n == 13:
            res = 13
        if n == 14:
            res = 14
        if n == 15:
            res = 15
        if n == 16:
            res = 16
    if c == 'B': # Blood and blood forming organs
        if n == 1:
            res = 17
        if n == 2:
            res = 18
        if n == 3:
            res = 19
        if n == 5:
            res = 20
        if n == 6:
            res = 21
    if c == 'C': # Cardiovascular system
        if n == 1:
            res = 22
        if n == 2:
            res = 23
        if n == 3:
            res = 24
        if n == 4:
            res = 25
        if n == 5:
            res = 26
        if n == 7:
            res = 27
        if n == 8:
            res = 28
        if n == 9:
            res = 29
        if n == 10:
            res = 30
    if c == 'D': # Dermatological drugs
        if n == 1:
            res = 31
        if n == 2:
            res = 32
        if n == 3:
            res = 33
        if n == 4:
            res = 34
        if n == 5:
            res = 35
        if n == 6:
            res = 36
        if n == 7:
            res = 37
        if n == 8:
            res = 38
        if n == 9:
            res = 39
        if n == 10:
            res = 40
        if n == 11:
            res = 41
    if c == 'G': # Genitourinary system and reproductive hormones
        if n == 1:
            res = 42
        if n == 2:
            res = 43
        if n == 3:
            res = 44
        if n == 4:
            res = 45
    if c == 'H': # Dermatological drugs
        if n == 1:
            res = 46
        if n == 2:
            res = 47
        if n == 3:
            res = 48
        if n == 4:
            res = 49
        if n == 5:
            res = 50
    if c == 'J': # Antiinfectives for systemic use
        if n == 1:
            res = 51
        if n == 2:
            res = 52
        if n == 4:
            res = 53
        if n == 5:
            res = 54
        if n == 6:
            res = 55
        if n == 7:
            res = 56
    if c == 'L': # Antineoplastic and immunomodulating agents
        if n == 1:
            res = 57
        if n == 2:
            res = 58
        if n == 3:
            res = 59
        if n == 4:
            res = 60
    if c == 'M': # Musculoskeletal system
        if n == 1:
            res = 61
        if n == 2:
            res = 62
        if n == 3:
            res = 63
        if n == 4:
            res = 64
        if n == 5:
            res = 65
        if n == 9:
            res = 66
    if c == 'N': # Nervous system
        if n == 1:
            res = 67
        if n == 2:
            res = 68
        if n == 3:
            res = 69
        if n == 4:
            res = 70
        if n == 5:
            res = 71
        if n == 6:
            res = 72
        if n == 7:
            res = 73
    if c == 'P': # Antiparasitic products, insecticides and repellents
        if n == 1:
            res = 74
        if n == 2:
            res = 75
        if n == 3:
            res = 76
    if c == 'R': # Respiratory system
        if n == 1:
            res = 77
        if n == 2:
            res = 78
        if n == 3:
            res = 79
        if n == 5:
            res = 80
        if n == 6:
            res = 81
        if n == 7:
            res = 82
    if c == 'S': # Sensory organs
        if n == 1:
            res = 83
        if n == 2:
            res = 84
        if n == 3:
            res = 85
    if c == 'V': # Dermatological drugs
        if n == 1:
            res = 86
        if n == 3:
            res = 87
        if n == 4:
            res = 88
        if n == 6:
            res = 89
        if n == 7:
            res = 90
        if n == 8:
            res = 91
        if n == 9:
            res = 92
        if n == 10:
            res = 93
        if n == 20:
            res = 94


    return res


def icd10_text(icd10):
    icd10 = icd10[:3]
    if icd10 == 'C7A':
        return "Malignant neuroendocrine tumors"
    if icd10 == 'C7B':
        return "Secondary neuroendocrine tumors"
    if icd10 == 'D3A':
        return "Benign neuroendocrine tumors"
    if icd10 == 'I5A': # included in I30-I5A
        return "heart disease"
    if icd10 == 'O9A': # included in O94-O9A
        return "obstetric conditions"
    if icd10 == 'Z3A':
        return "Persons encountering health services for examinations"

    c = icd10[0].upper()
    n = int(icd10[1:])
    # Certain infectious and parasitic diseases
    if c == 'A':
        if n >= 0 and n <= 9:
            res = "Intestinal infectious diseases"
        if n >= 15 and n <= 19:
            res = "Tuberculosis"
        if n >= 20 and n <= 28:
            res = "zoonotic bacterial diseases"
        if n >= 30 and n <= 49:
            res = "bacterial diseases"
        if n >= 50 and n <= 64:
            res = "Infections with a predominantly sexual mode of transmission"
        if n >= 65 and n <= 69:
            res = "spirochetal diseases"
        if n >= 70 and n <= 74:
            res = "diseases caused by chlamydiae"
        if n >= 75 and n <= 79:
            res = "Rickettsioses"
        if n >= 80 and n <= 89:
            res = "Viral and prion infections of the central nervous system"
        if n >= 90 and n <= 99:
            res = "Arthropod-borne viral fevers and viral hemorrhagic fevers"
    if c == 'B':
        if n >= 0 and n <= 9:
            res = "Viral infections characterized by skin and mucous membrane lesions"
        if n == 10:
            res = "human herpesviruses"
        if n >= 15 and n <= 19:
            res = "Viral hepatitis"
        if n >= 20 and n <= 24:
            res = "Human immunodeficiency virus disease"
        if n >= 25 and n <= 34:
            res = "Viral diseases"
        if n >= 35 and n <= 49:
            res = "Mycoses"
        if n >= 50 and n <= 64:
            res = "Protozoal diseases"
        if n >= 65 and n <= 84:
            res = "Helminthiases"
        if n >= 85 and n <= 89:
            res = "Pediculosis, acariasis and other infestations"
        if n >= 90 and n <= 94:
            res = "Sequelae of infectious and parasitic diseases"
        if n >= 95 and n <= 97:
            res = "Bacterial and viral infectious agents"
        if n >= 98 and n <= 99:
            res = "Other infectious diseases"
    # Neoplasms
    if c == 'C':
        if n >= 0 and n <= 14:
            res = "Malignant neoplasms of lip, oral cavity and pharynx"
        if n >= 15 and n <= 26:
            res = "Malignant neoplasms of digestive organs"
        if n >= 30 and n <= 39:
            res = "Malignant neoplasms of respiratory and intrathoracic organs"
        if n >= 40 and n <= 41:
            res = "Malignant neoplasms of bone and articular cartilage"
        if n >= 43 and n <= 44:
            res = "Melanoma and other malignant neoplasms of skin"
        if n >= 45 and n <= 49:
            res = "Malignant neoplasms of mesothelial and soft tissue"
        if n == 50:
            res = "Malignant neoplasms of breast"
        if n >= 51 and n <= 58:
            res = "Malignant neoplasms of female genital organs"
        if n >= 60 and n <= 63:
            res = "Malignant neoplasms of male genital organs"
        if n >= 64 and n <= 68:
            res = "Malignant neoplasms of urinary tract"
        if n >= 69 and n <= 72:
            res = "Malignant neoplasms of eye, brain and other parts of central nervous system"
        if n >= 73 and n <= 75:
            res = "Malignant neoplasms of thyroid and other endocrine glands"
        if n >= 76 and n <= 80:
            res = "Malignant neoplasms of ill-defined, other secondary and unspecified sites"
        if n >= 81 and n <= 96:
            res = "Malignant neoplasms of lymphoid, hematopoietic and related tissue"
        if n == 97:
            res = "Malignant neoplasms of independent (primary) multiple sites"
    if c == 'D':
        if n >= 0 and n <= 9:
            res = "In situ neoplasms"
        if n >= 10 and n <= 36:
            res = "Benign neoplasms"
        if n >= 37 and n <= 48:
            res = "Neoplasms of uncertain behavior, polycythemia vera and myelodysplastic syndromes"
        if n == 49:
            res = "Neoplasms of unspecified behavior"
    # Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism
        if n >= 50 and n <= 53:
            res = "Nutritional anemias"
        if n >= 55 and n <= 59:
            res = "Hemolytic anemias"
        if n >= 60 and n <= 64:
            res = "Aplastic and other anemias and other bone marrow failure syndromes"
        if n >= 65 and n <= 69:
            res = "Coagulation defects, purpura and other hemorrhagic conditions"
        if n >= 70 and n <= 77:
            res = "disorders of blood and blood-forming organs"
        if n == 78:
            res = "Intraoperative and postprocedural complications of the spleen"
        if n >= 80 and n <= 89:
            res = "Certain disorders involving the immune mechanism"
    # Endocrine, nutritional and metabolic diseases
    if c == 'E':
        if n >= 0 and n <= 7:
            res = "Disorders of thyroid gland"
        if n >= 8 and n <= 14:
            res = "Diabetes mellitus"
        if n >= 15 and n <= 16:
            res = "Other disorders of glucose regulation and pancreatic internal secretion"
        if n >= 20 and n <= 35:
            res = "Disorders of other endocrine glands"
        if n == 36:
            res = "Intraoperative complications of endocrine system"
        if n >= 40 and n <= 46:
            res = "Malnutrition"
        if n >= 50 and n <= 64:
            res = "Other nutritional deficiencies"
        if n >= 65 and n <= 68:
            res = "Overweight, obesity and other hyperalimentation"
        if n >= 70 and n <= 88:
            res = "Metabolic disorders"
        if n == 89:
            res = "Postprocedural endocrine and metabolic complications"
        if n == 90:
            res = "Nutritional and metabolic disorders" # edited
    # Mental, Behavioral and Neurodevelopmental disorders
    if c == 'F':
        if n >= 0 and n <= 9:
            res = "Mental disorders"
        if n >= 10 and n <= 19:
            res = "Mental and behavioral disorders"
        if n >= 20 and n <= 29:
            res = "Schizophrenia, schizotypal, delusional, and other non-mood psychotic disorders"
        if n >= 30 and n <= 39:
            res = "Mood disorders"
        if n >= 40 and n <= 48:
            res = "Anxiety, dissociative, stress-related, somatoform and other nonpsychotic mental disorders"
        if n >= 50 and n <= 59:
            res = "Behavioral syndromes associated with physiological disturbances and physical factors"
        if n >= 60 and n <= 69:
            res = "Disorders of adult personality and behavior"
        if n >= 70 and n <= 79:
            res = "Intellectual disabilities"
        if n >= 80 and n <= 89:
            res = "Pervasive and specific developmental disorders"
        if n >= 90 and n <= 98:
            res = "Behavioral and emotional disorders"
        if n == 99:
            res = "mental disorder"
    # Diseases of the nervous system
    if c == 'G':
        if n >= 0 and n <= 9:
            res = "Inflammatory diseases of the central nervous system"
        if n >= 10 and n <= 14:
            res = "Systemic atrophies primarily affecting the central nervous system"
        if n >= 20 and n <= 26:
            res = "Extrapyramidal and movement disorders"
        if n >= 30 and n <= 32:
            res = "degenerative diseases of the nervous system"
        if n >= 35 and n <= 37:
            res = "Demyelinating diseases of the central nervous system"
        if n >= 40 and n <= 47:
            res = "Episodic and paroxysmal disorders"
        if n >= 50 and n <= 59:
            res = "Nerve, nerve root and plexus disorders"
        if n >= 60 and n <= 65:
            res = "Polyneuropathies and disorders of the peripheral nervous system"
        if n >= 70 and n <= 73:
            res = "Diseases of myoneural junction and muscle"
        if n >= 80 and n <= 83:
            res = "Cerebral palsy and paralytic syndromes"
        if n >= 89 and n <= 99:
            res = "disorders of the nervous system"
    # Diseases of the eye and adnexa
    if c == 'H':
        if n >= 0 and n <= 6:
            res = "Disorders of eyelid, lacrimal system and orbit"
        if n >= 10 and n <= 13:
            res = "Disorders of conjunctiva"
        if n >= 15 and n <= 22:
            res = "Disorders of sclera, cornea, iris and ciliary body"
        if n >= 25 and n <= 28:
            res = "Disorders of lens"
        if n >= 30 and n <= 36:
            res = "Disorders of choroid and retina"
        if n >= 40 and n <= 42:
            res = "Glaucoma"
        if n >= 43 and n <= 45:
            res = "Disorders of vitreous body and globe"
        if n >= 46 and n <= 48:
            res = "Disorders of optic nerve and visual pathways"
        if n >= 49 and n <= 52:
            res = "Disorders of ocular muscles, binocular movement, accommodation and refraction"
        if n >= 53 and n <= 54:
            res = "Visual disturbances and blindness"
        if n >= 55 and n <= 57:
            res = "disorders of eye and adnexa"
        if n == 59:
            res = "Intraoperative and postprocedural complications and disorders of eye and adnexa"
    # Diseases of the ear and mastoid process
        if n >= 60 and n <= 62:
            res = "Diseases of external ear"
        if n >= 65 and n <= 75:
            res = "Diseases of middle ear and mastoid"
        if n >= 80 and n <= 83:
            res = "Diseases of inner ear"
        if n >= 90 and n <= 94:
            res = "disorders of ear"
        if n == 95:
            res = "Intraoperative and postprocedural complications and disorders of ear and mastoid process"
    # Diseases of the circulatory system
    if c == 'I':
        if n >= 0 and n <= 2:
            res = "Acute rheumatic fever"
        if n >= 5 and n <= 9:
            res = "Chronic rheumatic heart diseases"
        if n >= 10 and n <= 16:
            res = "Hypertensive diseases"
        if n >= 20 and n <= 25:
            res = "Ischemic heart diseases"
        if n >= 26 and n <= 28:
            res = "Pulmonary heart disease and diseases of pulmonary circulation"
        if n >= 30 and n <= 52:
            res = "heart disease"
        if n >= 60 and n <= 69:
            res = "Cerebrovascular diseases"
        if n >= 70 and n <= 79:
            res = "Diseases of arteries, arterioles and capillaries"
        if n >= 80 and n <= 89:
            res = "Diseases of veins, lymphatic vessels and lymph nodes, not elsewhere classified"
        if n >= 95 and n <= 99:
            res = "Other and unspecified disorders of the circulatory system"
    # Diseases of the respiratory system
    if c == 'J':
        if n >= 0 and n <= 6:
            res = "Acute upper respiratory infections"
        if n >= 9 and n <= 18:
            res = "Influenza and pneumonia"
        if n >= 20 and n <= 22:
            res = "acute lower respiratory infections"
        if n >= 30 and n <= 39:
            res = "diseases of upper respiratory tract"
        if n >= 40 and n <= 47:
            res = "Chronic lower respiratory diseases"
        if n >= 60 and n <= 70:
            res = "Lung diseases due to external agents"
        if n >= 80 and n <= 84:
            res = "respiratory diseases principally affecting the interstitium"
        if n >= 85 and n <= 86:
            res = "Suppurative and necrotic conditions of the lower respiratory tract"
        if n >= 90 and n <= 94:
            res = "diseases of the pleura"
        if n == 95:
            res = "Intraoperative and postprocedural complications and disorders of respiratory system"
        if n >= 96 and n <= 99:
            res = "diseases of the respiratory system"
    # Diseases of the digestive system
    if c == 'K':
        if n >= 0 and n <= 14:
            res = "Diseases of oral cavity and salivary glands"
        if n >= 20 and n <= 31:
            res = "Diseases of esophagus, stomach and duodenum"
        if n >= 35 and n <= 38:
            res = "Diseases of appendix"
        if n >= 40 and n <= 46:
            res = "Hernia"
        if n >= 50 and n <= 52:
            res = "Noninfective enteritis and colitis"
        if n >= 55 and n <= 64:
            res = "diseases of intestines"
        if n >= 65 and n <= 68:
            res = "Diseases of peritoneum and retroperitoneum"
        if n >= 70 and n <= 77:
            res = "Diseases of liver"
        if n >= 80 and n <= 87:
            res = "Disorders of gallbladder, biliary tract and pancreas"
        if n >= 90 and n <= 95:
            res = "diseases of the digestive system"
    # Diseases of the skin and subcutaneous tissue
    if c == 'L':
        if n >= 0 and n <= 8:
            res = "Infections of the skin and subcutaneous tissue"
        if n >= 10 and n <= 14:
            res = "Bullous disorders"
        if n >= 20 and n <= 30:
            res = "Dermatitis and eczema"
        if n >= 40 and n <= 45:
            res = "Papulosquamous disorders"
        if n >= 49 and n <= 54:
            res = "Urticaria and erythema"
        if n >= 55 and n <= 59:
            res = "Radiation-related disorders of the skin and subcutaneous tissue"
        if n >= 60 and n <= 75:
            res = "Disorders of skin appendages"
        if n == 76:
            res = "Intraoperative and postprocedural complications of skin and subcutaneous tissue"
        if n >= 80 and n <= 99:
            res = "Other disorders of the skin and subcutaneous tissue"
    # Diseases of the musculoskeletal system and connective tissue
    if c == 'M':
        if n >= 0 and n <= 2:
            res = "Infectious arthropathies"
        if n == 4:
            res = "Autoinflammatory syndromes"
        if n >= 5 and n <= 14:
            res = "Inflammatory polyarthropathies"
        if n >= 15 and n <= 19:
            res = "Osteoarthritis"
        if n >= 20 and n <= 25:
            res = "joint disorders"
        if n >= 26 and n <= 27:
            res = "Dentofacial anomalies and other disorders of jaw"
        if n >= 30 and n <= 36:
            res = "Systemic connective tissue disorders"
        if n >= 40 and n <= 43:
            res = "Deforming dorsopathies"
        if n >= 45 and n <= 49:
            res = "Spondylopathies"
        if n >= 50 and n <= 54:
            res = "dorsopathies"
        if n >= 60 and n <= 64:
            res = "Disorders of muscles"
        if n >= 65 and n <= 69:
            res = "Disorders of synovium and tendon"
        if n >= 70 and n <= 79:
            res = "soft tissue disorders"
        if n >= 80 and n <= 85:
            res = "Disorders of bone density and structure"
        if n >= 86 and n <= 90:
            res = "osteopathies"
        if n >= 91 and n <= 94:
            res = "Chondropathies"
        if n == 95:
            res = "disorders of the musculoskeletal system and connective tissue"
        if n == 96:
            res = "Intraoperative and postprocedural complications and disorders of musculoskeletal system"
        if n == 97:
            res = "Periprosthetic fracture around internal prosthetic joint"
        if n == 99:
            res = "Biomechanical lesions, not elsewhere classified"
    # Diseases of the genitourinary system
    if c == 'N':
        if n >= 0 and n <= 8:
            res = "Glomerular diseases"
        if n >= 10 and n <= 16:
            res = "Renal tubulo-interstitial diseases"
        if n >= 17 and n <= 19:
            res = "Acute kidney failure and chronic kidney disease"
        if n >= 20 and n <= 23:
            res = "Urolithiasis"
        if n >= 25 and n <= 29:
            res = "disorders of kidney and ureter"
        if n >= 30 and n <= 39:
            res = "diseases of the urinary system"
        if n >= 40 and n <= 53:
            res = "Diseases of male genital organs"
        if n >= 60 and n <= 65:
            res = "Disorders of breast"
        if n >= 70 and n <= 77:
            res = "Inflammatory diseases of female pelvic organs"
        if n >= 80 and n <= 98:
            res = "Noninflammatory disorders of female genital tract"
        if n == 99:
            res = "Intraoperative and postprocedural complications and disorders of genitourinary system"
    # Pregnancy, childbirth and the puerperium
    if c == 'O':
        if n >= 0 and n <= 8:
            res = "Pregnancy with abortive outcome"
        if n == 9:
            res = "Supervision of high risk pregnancy"
        if n >= 10 and n <= 16:
            res = "Edema, proteinuria and hypertensive disorders in pregnancy, childbirth and the puerperium"
        if n >= 20 and n <= 29:
            res = "maternal disorders predominantly related to pregnancy"
        if n >= 30 and n <= 48:
            res = "Maternal care related to the fetus and amniotic cavity and possible delivery problem"
        if n >= 60 and n <= 77:
            res = "Complications of labor and delivery"
        if n >= 80 and n <= 84:
            res = "Encounter for delivery"
        if n >= 85 and n <= 92:
            res = "Complications predominantly related to the puerperium"
        if n >= 94 and n <= 99:
            res = "obstetric conditions, not elsewhere classified"
    # Certain conditions originating in the perinatal period
    if c == 'P':
        if n >= 0 and n <= 4:
            res = "Newborn affected by maternal factors and by complications of pregnancy, labor, and delivery"
        if n >= 5 and n <= 8:
            res = "Disorders of newborn related to length of gestation and fetal growth"
        if n == 9:
            res = "Abnormal findings on neonatal screening"
        if n >= 10 and n <= 15:
            res = "Birth trauma"
        if n >= 19 and n <= 29:
            res = "Respiratory and cardiovascular disorders specific to the perinatal period"
        if n >= 35 and n <= 39:
            res = "Infections specific to the perinatal period"
        if n >= 50 and n <= 61:
            res = "Hemorrhagic and hematological disorders of newborn"
        if n >= 70 and n <= 74:
            res = "Transitory endocrine and metabolic disorders specific to newborn"
        if n >= 75 and n <= 78:
            res = "Digestive system disorders of newborn"
        if n >= 80 and n <= 83:
            res = "Conditions involving the integument and temperature regulation of newborn"
        if n == 84:
            res = "problems with newborn"
        if n >= 90 and n <= 96:
            res = "disorders originating in the perinatal period"
    # Congenital malformations, deformations and chromosomal abnormalities
    if c == 'Q':
        if n >= 0 and n <= 7:
            res = "Congenital malformations of the nervous system"
        if n >= 10 and n <= 18:
            res = "Congenital malformations of eye, ear, face and neck"
        if n >= 20 and n <= 28:
            res = "Congenital malformations of the circulatory system"
        if n >= 30 and n <= 34:
            res = "Congenital malformations of the respiratory system"
        if n >= 35 and n <= 37:
            res = "Cleft lip and cleft palate"
        if n >= 38 and n <= 45:
            res = "congenital malformations of the digestive system"
        if n >= 50 and n <= 56:
            res = "Congenital malformations of genital organs"
        if n >= 60 and n <= 64:
            res = "Congenital malformations of the urinary system"
        if n >= 65 and n <= 79:
            res = "Congenital malformations and deformations of the musculoskeletal system"
        if n >= 80 and n <= 89:
            res = "congenital malformations"
        if n >= 90 and n <= 99:
            res = "Chromosomal abnormalities, not elsewhere classified"
    # Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified
    if c == 'R':
        if n >= 0 and n <= 9:
            res = "Symptoms and signs involving the circulatory and respiratory systems"
        if n >= 10 and n <= 19:
            res = "Symptoms and signs involving the digestive system and abdomen"
        if n >= 20 and n <= 23:
            res = "Symptoms and signs involving the skin and subcutaneous tissue"
        if n >= 25 and n <= 29:
            res = "Symptoms and signs involving the nervous and musculoskeletal systems"
        if n >= 30 and n <= 39:
            res = "Symptoms and signs involving the genitourinary system"
        if n >= 40 and n <= 46:
            res = "Symptoms and signs involving cognition, perception, emotional state and behavior"
        if n >= 47 and n <= 49:
            res = "Symptoms and signs involving speech and voice"
        if n >= 50 and n <= 69:
            res = "General symptoms and signs"
        if n >= 70 and n <= 79:
            res = "Abnormal findings on examination of blood, without diagnosis"
        if n >= 80 and n <= 82:
            res = "Abnormal findings on examination of urine, without diagnosis"
        if n >= 83 and n <= 89:
            res = "Abnormal findings on examination of other body fluids, substances and tissues, without diagnosis"
        if n >= 90 and n <= 94:
            res = "Abnormal findings on diagnostic imaging and in function studies, without diagnosis"
        if n == 97:
            res = "Abnormal tumor markers"
        if n == 99 or n == 95 or n == 96 or n == 98:
            res = "Ill-defined and unknown cause of mortality"
    # Injury, poisoning and certain other consequences of external causes
    if c == 'S':
        if n >= 0 and n <= 9:
            res = "Injuries to the head"
        if n >= 10 and n <= 19:
            res = "Injuries to the neck"
        if n >= 20 and n <= 29:
            res = "Injuries to the thorax"
        if n >= 30 and n <= 39:
            res = "Injuries to the abdomen, lower back, lumbar spine, pelvis and external genitals"
        if n >= 40 and n <= 49:
            res = "Injuries to the shoulder and upper arm"
        if n >= 50 and n <= 59:
            res = "Injuries to the elbow and forearm"
        if n >= 60 and n <= 69:
            res = "Injuries to the wrist, hand and fingers"
        if n >= 70 and n <= 79:
            res = "Injuries to the hip and thigh"
        if n >= 80 and n <= 89:
            res = "Injuries to the knee and lower leg"
        if n >= 90 and n <= 99:
            res = "Injuries to the ankle and foot"
    if c == 'T':
        if n >= 0 and n <= 7:
            res = "Injuries involving multiple body regions"
        if n >= 8 and n <= 14:
            res = "Injury of unspecified body region"
        if n >= 15 and n <= 19:
            res = "Effects of foreign body entering through natural orifice"
        if n >= 20 and n <= 25:
            res = "Burns and corrosions of external body surface"
        if n >= 26 and n <= 28:
            res = "Burns and corrosions confined to eye and internal organs"
        if n >= 29 and n <= 32:
            res = "Burns and corrosions of multiple and unspecified body regions"
        if n >= 33 and n <= 35:
            res = "Frostbite"
        if n >= 36 and n <= 50:
            res = "Poisoning by, adverse effect of and underdosing of drugs, medicaments and biological substances"
        if n >= 51 and n <= 65:
            res = "Toxic effects of substances chiefly nonmedicinal as to source"
        if n >= 66 and n <= 78:
            res = "effects of external causes"
        if n == 79:
            res = "Certain early complications of trauma"
        if n >= 80 and n <= 88:
            res = "Complications of surgical and medical care"
        if n >= 90 and n <= 98:
            res = "Sequelae of injuries, of poisoning"
    # Codes for special purposes
    if c == 'U':
        if n >= 0 and n <= 49:
            res = "Provisional assignment of new diseases of uncertain etiology or emergency use"
        if n >= 50 and n <= 89:
            res = "Resistance to antimicrobial and antineoplastic drugs  "
    # External causes of morbidity
    if c == 'V':
        if n >= 0 and n <= 9:
            res = "Pedestrian injured in transport accident"
        if n >= 10 and n <= 19:
            res = "Pedal cycle rider injured in transport accident"
        if n >= 20 and n <= 29:
            res = "Motorcycle rider injured in transport accident"
        if n >= 30 and n <= 39:
            res = "Occupant of three-wheeled motor vehicle injured in transport accident"
        if n >= 40 and n <= 49:
            res = "Car occupant injured in transport accident"
        if n >= 50 and n <= 59:
            res = "Occupant of pick-up truck or van injured in transport accident"
        if n >= 60 and n <= 69:
            res = "Occupant of heavy transport vehicle injured in transport accident"
        if n >= 70 and n <= 79:
            res = "Bus occupant injured in transport accident"
        if n >= 80 and n <= 89:
            res = "Other land transport accidents"
        if n >= 90 and n <= 94:
            res = "Water transport accidents"
        if n >= 95 and n <= 97:
            res = "Air and space transport accidents"
        if n >= 98 and n <= 99:
            res = "Other and unspecified transport accidents"
    if c == 'W':
        if n >= 0 and n <= 19:
            res = "Slipping, tripping, stumbling and falls"
        if n >= 20 and n <= 49:
            res = "Exposure to inanimate mechanical forces"
        if n >= 50 and n <= 64:
            res = "Exposure to animate mechanical forces"
        if n >= 65 and n <= 74:
            res = "Accidental non-transport drowning and submersion"
        if n >= 85 and n <= 99:
            res = "Exposure to electric current, radiation and extreme ambient air temperature and pressure"
    if c == 'X':
        if n >= 0 and n <= 8:
            res = "Exposure to smoke, fire and flames"
        if n >= 10 and n <= 19:
            res = "Contact with heat and hot substances"
        if n >= 30 and n <= 39:
            res = "Exposure to forces of nature"
        if n >= 40 and n <= 49:
            res = "Overexertion and strenuous or repetitive movements"
        if n >= 50 and n <= 58:
            res = "Accidental exposure to other specified factors"
        if n >= 60 and n <= 84:
            res = "Intentional self-harm"
        if n >= 92 and n <= 99:
            res = "Assault"
    if c == 'Y':
        if n >= 0 and n <= 9:
            res = "Assault"
        if n >= 10 and n <= 34:
            res = "Event of undetermined intent"
        if n >= 35 and n <= 38:
            res = "Legal intervention, operations of war, military operations, and terrorism"
        if n >= 40 and n <= 69:
            res = "Misadventures to patients during surgical and medical care"
        if n >= 70 and n <= 82:
            res = "Medical devices associated with adverse incidents in diagnostic and therapeutic use"
        if n >= 83 and n <= 84:
            res = "Surgical and other medical procedures as the cause of abnormal reaction "
        if n >= 85 and n <= 99:
            res = "Supplementary factors related to causes of morbidity"
    # Factors influencing health status and contact with health services
    if c == 'Z':
        if n >= 0 and n <= 13:
            res = "Persons encountering health services for examinations"
        if n >= 14 and n <= 15:
            res = "Genetic carrier and genetic susceptibility to disease"
        if n == 16:
            res = "Resistance to antimicrobial drugs"
        if n == 17:
            res = "Estrogen receptor status"
        if n == 18:
            res = "Retained foreign body fragments"
        if n == 19:
            res = "Hormone sensitivity malignancy status"
        if n >= 20 and n <= 29:
            res = "Persons with potential health hazards related to communicable diseases"
        if n >= 30 and n <= 39:
            res = "Persons encountering health services in circumstances related to reproduction"
        if n >= 40 and n <= 54:
            res = "Encounters for other specific health care"
        if n >= 55 and n <= 65:
            res = "Persons with potential health hazards related to socioeconomic and psychosocial circumstances"
        if n == 66:
            res = "Do not resuscitate status"
        if n == 67:
            res = "Blood type"
        if n == 68:
            res = "Body mass index"
        if n >= 69 and n <= 76:
            res = "Persons encountering health services in other circumstances"
        if n >= 77 and n <= 99:
            res = "Persons with potential health hazards related to family and personal history"

    try:
        return res
    except:
        return 'Unknown'


def infer_chapter_from_code(code_norm: str) -> str:
    import re
    # code_norm like 'S20.0' or 'S200' etc; take letter + first two digits
    if not code_norm:
        return "Unknown chapter"
    letter = code_norm[0]
    m = re.match(r"^([A-Z])(\d{2})", code_norm)
    num = int(m.group(2)) if m else None

    # Special purposes
    if letter == "U":
        return "Codes for special purposes (U00–U85)"

    # H split
    if letter == "H" and num is not None:
        if 0 <= num <= 59:
            return "Diseases of the eye and adnexa"
        if 60 <= num <= 95:
            return "Diseases of the ear and mastoid process"

    # Single-letter chapters by WHO/ICD-10 groupings
    if letter in ["A", "B"]:
        return "Certain infectious and parasitic diseases"
    if letter == "C":
        return "Neoplasms (C00–D49)"
    if letter == "D" and num is not None:
        # D00–D49 also neoplasms; D50–D89 blood/immune
        if 0 <= num <= 49:
            return "Neoplasms"
        if 50 <= num <= 89:
            return "Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism"
    if letter == "E":
        return "Endocrine, nutritional and metabolic diseases"
    if letter == "F":
        return "Mental, behavioral and neurodevelopmental disorders"
    if letter == "G":
        return "Diseases of the nervous system"
    if letter == "I":
        return "Diseases of the circulatory system"
    if letter == "J":
        return "Diseases of the respiratory system"
    if letter == "K":
        return "Diseases of the digestive system"
    if letter == "L":
        return "Diseases of the skin and subcutaneous tissue"
    if letter == "M":
        return "Diseases of the musculoskeletal system and connective tissue"
    if letter == "N":
        return "Diseases of the genitourinary system"
    if letter == "O":
        return "Pregnancy, childbirth and the puerperium"
    if letter == "P":
        return "Certain conditions originating in the perinatal period"
    if letter == "Q":
        return "Congenital malformations, deformations and chromosomal abnormalities"
    if letter == "R":
        return "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified"
    if letter in ["S", "T"]:
        return "Injury, poisoning and certain other consequences of external causes"
    if letter in ["V", "W", "X", "Y"]:
        return "External causes of morbidity"
    if letter == "Z":
        return "Factors influencing health status and contact with health services"

    return "Unknown chapter"
