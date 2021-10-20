import pandas as pd
import numpy as np
from datetime import datetime
import os
from selected_features import PATIENT_FEATURES, ORGAN_FEATURES, INITIAL_PATIENT_FEATURES, substitution_pairs

liver_data_column_definition = './Raw Data/Transplant Data/LIVER_DATA.htm'
liver_data = './Raw Data/Transplant Data/LIVER_DATA.DAT'

with open(liver_data_column_definition, 'rb') as html_table:
    tables = pd.read_html(html_table, index_col=0)
    df = tables[0]
    columns = df.LABEL.to_list()

with open(liver_data, 'rb') as dat_table:
    liver_records = pd.read_table(dat_table,
                                  names=columns,
                                  na_values=[' ', '.', 'Unknown', '**'],
                                  encoding='cp1252')

#Removing cases that are not used in this project

print('original dataset entries:%d' % len(liver_records))
#remove all cases do not have a waitlist ID code
liver = liver_records.dropna(subset=['WL_ID_CODE', 'INIT_DATE'])
#after removal of these cases we are left with this many rows
print('entries after drop cases without WL_ID_CODE and INIT_DATE:%d' %
      len(liver))

liver.loc[:, 'INIT_DATE'] = pd.to_datetime(liver['INIT_DATE'])

#keep all patients joined waitlist after year 2002
liver = liver.loc[liver['INIT_DATE'].dt.year >= 2002]

#remove all pediatric transplants
liver = liver.loc[liver['INIT_AGE'] > 18]

#remove all living donor transplants
liver = liver.loc[liver['DON_TY'] != 'L']

#Can consider removing multiorgan transplants as other models have excluded these. For now
#they will be removed
liver = liver.loc[liver['MULTIORG'] != 'Y']

# Remove patients been retransplanted
liver = liver.loc[liver['PREV_TX_ANY'] != 'Y']

print('entries after drop some unwanted cases:%d' % len(liver))

#variable pertaining to post-transplantation and therefore should not be used as part of the
#model

post_outcomes = [
    'ACUTE_REJ_EPI', 'BILIARY', 'COD', 'COD_OSTXT', 'COD_OSTXT_WL', 'COD2',
    'COD2_OSTXT', 'COD3', 'COD3_OSTXT', 'COD_WL', 'DIFFUSE_CHOLANG',
    'DIS_ALKPHOS', 'DIS_SGOT', 'DISCHARGE_DATE', 'FUNC_STAT_TRF',
    'GRF_FAIL_CAUSE_OSTXT', 'GRF_FAIL_DATE', 'GRF_STAT', 'GSTATUS', 'GTIME',
    'HEP_DENOVO', 'HEP_RECUR', 'HEPATIC_ART_THROM', 'HEPATIC_OUT_OBS',
    'INFECT', 'LOS', 'OTHER_VASC_THROMB', 'PORTAL_VEIN_THROM', 'PRI_GRF_FAIL',
    'PRI_NON_FUNC', 'PX_NON_COMPL', 'RECUR_DISEASE', 'REJ_ACUTE',
    'REJ_CHRONIC', 'TRTREJ1Y', 'TRTREJ6M', 'VASC_THROMB'
]

liverdrop = liver.drop(post_outcomes, axis=1)

#These variable only pertain to living donors which we are excluding from analysis
living_donor = [
    'PRIV_INS_DON', 'PRI_PAYMENT_DON', 'PRI_PAYMENT_CTRY_DON', 'MEDICARE_DON',
    'MEDICAID_DON', 'LIV_DON_TY', 'LIV_DON_TY_OSTXT', 'HMO_PPO_DON',
    'HCV_TEST_DON', 'HCV_ANTIBODY_DON', 'HCV_RIBA_DON', 'HCV_RNA_DON',
    'HBV_TEST_DON', 'HBV_DNA_DON', 'FREE_DON', 'EDUCATION_DON', 'EBV_TEST_DON',
    'DONATION_DON', 'DON_TY', 'CMV_OLD_LIV_DON', 'CMV_TEST_DON', 'CMV_IGG_DON',
    'CMV_IGM_DON', 'OTH_GOVT_DON', 'SELF_DON', 'STATUS_LDR'
]

liverdrop = liverdrop.drop(living_donor, axis=1)

#These variables are no longer collected after 2004
old_variables = [
    'PREV_PI_TX_TCR_ARCHIVE', 'PGE', 'OTH_DON_MED1_OSTXT_DON_OLD',
    'OTH_DON_MED2_OSTXT_DON_OLD', 'OTH_DON_MED3_OSTXT_DON_OLD', 'MRCREATG_OLD',
    'IABP', 'HEPD_OLD', 'HBEAB_OLD', 'ENCEPH_TCR', 'ENCEPH_TRR_OLD',
    'CONTIN_ALCOHOL_OLD_DON', 'CONTIN_IV_DRUG_OLD_DON', 'CLSTR_OLD',
    'CLSTRTYP_OLD', 'ASCITES_TCR', 'ASCITES_TRR_OLD', 'VAD_TAH',
    'MUSCLE_WAST_TCR', 'INOTROPES', 'DOBUT_DON_OLD', 'DOPAMINE_DON_OLD',
    'HTLV1_OLD_DON', 'HTLV2_OLD_DON', 'PRETREAT_MED_DON_OLD',
    'TRANSFUS_INTRAOP_NUM_OLD_DON', 'TRANSFUS_PRIOR_NUM_OLD_DON', 'ECMO'
]

liverdrop = liverdrop.drop(old_variables, axis=1)

#These variables have either zero variability, or nearly zero variability
no_variability = [
    'TX_MELD', 'INIT_MELD_OR_PELD', 'FINAL_MELD_OR_PELD', 'DATA_WAITLIST',
    'DATA_TRANSPLANT', 'AGE_GROUP', 'RECOV_COUNTRY', 'PRI_PAYMENT_CTRY_TCR',
    'PRI_PAYMENT_CTRY_TRR', 'ARTIFICIAL_LI_TCR', 'ARTIFICIAL_LI_TRR'
]

liverdrop = liverdrop.drop(no_variability, axis=1)

#variables with no observations
no_obs = [
    'TOT_SERUM_ALBUM', 'FINAL_CTP_SCORE', 'OTH_LIFE_SUP_OSTXT_TCR',
    'HIST_ALCOHOL_OLD_DON', 'CMV_NUCLEIC_DON', 'LT_ONE_WEEK_DON',
    'HIST_IV_DRUG_OLD_DON'
]

#pertaining to dates that are either unimportant, or can be extracted from other data
dates = [
    'LIST_MELD', 'PX_STAT_DATE', 'PREV_TX_DATE', 'VAL_DT_DDR', 'VAL_DT_LDR',
    'VAL_DT_TCR', 'VAL_DT_TRR', 'TX_YEAR', 'RETXDATE', 'REFERRAL_DATE'
]

#Consider deriving new features from these data, i.e. recovery-admit
#dates1=['RECOVERY_DATE_DON', 'ADMISSION_DATE', 'ADMIT_DATE_DON', TX_DATE, 'INIT_DATE']
#liverdrop=liverdrop.drop(dates1,axis=1)

#pertaining to waitlist only
wait = ['REM_CD']

liverdrop = liverdrop.drop(no_obs, axis=1)

liverdrop = liverdrop.drop(dates, axis=1)

liverdrop = liverdrop.drop(wait, axis=1)

#These variables have extreme missingness and are almost certainty irrelevant
missing = [
    'INIT_CTP_SCORE', 'ACADEMIC_LEVEL_TCR', 'ACADEMIC_LEVEL_TRR',
    'ACADEMIC_PRG_TCR', 'ACADEMIC_PRG_TRR', 'COD_OSTXT_DON',
    'PULM_INF_CONF_DON', 'OTHER_INF_CONF_DON', 'CORE_COOL_DON'
]

liverdrop = liverdrop.drop(missing, axis=1)

#duplicate infromation
duplicate = ['LISTYR']

liverdrop = liverdrop.drop(duplicate, axis=1)

#admin
admin = ['STATUS_DDR', 'STATUS_TCR', 'STATUS_TRR']

liverdrop = liverdrop.drop(admin, axis=1)

#these are a list of identifier codes that likely will be dropped. The only code I will not drop now is
#'WL_ID_CODE' as we may use this to link to the liver waitlist csv file at a later time.
#is there any utility in adding such codes into the model (i.e. where a patient was listed)
#whether the starting center is different from the end center?

identifier = [
    'TRR_ID_CODE', 'INIT_OPO_CTR_CODE', 'END_OPO_CTR_CODE', 'OPO_CTR_CODE',
    'CTR_CODE', 'PT_CODE'
]

liverdrop = liverdrop.drop(identifier, axis=1)

#These variables have high missingness and are likely irrelevant
missing_unimp = [
    'CONTIN_CIG_DON', 'CONTIN_COCAINE_DON', 'CONTIN_OTH_DRUG_DON', 'DIET_DON',
    'DIURETICS_DON', 'VESSELS_NUM_STEN_DON', 'OTHER_INF_DON'
]

#Consider addming drmis, hlamis, hbsab_don to this list

liverdrop = liverdrop.drop(missing_unimp, axis=1)

#The variables have few positives and are almost certainly not important
lowpos = [
    'CANCER_FREE_INT_DON', 'CANCER_OTH_OSTXT_DON', 'CANCER_SITE_DON',
    'LV_EJECT_DON', 'LV_EJECT_METH_DON', 'EXTRACRANIAL_CANCER_DON',
    'CITIZEN_COUNTRY', 'HBV_SUR_ANTIGEN_DON'
]

liverdrop = liverdrop.drop(lowpos, axis=1)

#These data are nearly duplicates to other variable. not sure why there is some mismatch
#could consider running model with these variables instead of the other or both
duplicates1 = [
    'FINAL_ALBUMIN', 'FINAL_ASCITES', 'FINAL_ENCEPH',
    'FINAL_DIALYSIS_PRIOR_WEEK', 'FINAL_SERUM_CREAT',
    'FINAL_MELD_PELD_LAB_SCORE', 'FINAL_BILIRUBIN', 'TX_PROCEDUR_TY',
    'YR_ENTRY_US_TCR', 'CONTROLLED_DON', 'PX_STAT', 'MALIG'
]

liverdrop = liverdrop.drop(duplicates1, axis=1)

#Sparse data and almost certainly unimportant
sparse = [
    'INO_PROCURE_OSTXT_1', 'INO_PROCURE_OSTXT_2', 'INO_PROCURE_OSTXT_3',
    'PT_OTH1_OSTXT_DON', 'PT_OTH2_OSTXT_DON', 'PT_OTH3_OSTXT_DON',
    'PT_OTH4_OSTXT_DON', 'OTHER_INF_OSTXT_DON', 'DGN_OSTXT_TCR',
    'DGN2_OSTXT_TCR', 'EXC_OTHER_DIAG', 'DIAG_OSTXT', 'PREV_MALIG_TY_OSTXT',
    'MALIG_TY_OSTXT', 'MALIG_OSTXT_TRR', 'PERM_STATE', 'PERM_STATE_TRR',
    'HOME_STATE_DON'
]

liverdrop = liverdrop.drop(sparse, axis=1)

#Removing variables related to multiorgan transplant
multi = [
    'MULTIORG', 'WLHL', 'WLHR', 'WLIN', 'WLKI', 'WLKP', 'WLLI', 'WLLU', 'WLPA',
    'WLPI', 'WLVC', 'TXHRT', 'TXINT', 'TXKID', 'TXLNG', 'TXPAN', 'TXVCA'
]

liverdrop = liverdrop.drop(multi, axis=1)

#hla loci
#many levels per variable, previously has been shown not to be important for liver
#remove for now. also high missingness

locus = [
    'DDR1',
    'DDR2',
    'DA1',
    'DA2',
    'DB1',
    'DB2',
    'RA1',
    'RA2',
    'RB1',
    'RB2',
    'RDR1',
    'RDR2',
    'DB1',
    'DB2',
]
liverdrop = liverdrop.drop(locus, axis=1)

#meaningless
meaningless = ['PT_OTH_DON', 'OTH_LIFE_SUP_TCR', 'OTH_LIFE_SUP_TRR']
liverdrop = liverdrop.drop(meaningless, axis=1)

#donor unimmportant
donorunim = [
    'DIABDUR_DON', 'HYPERTENS_DUR_DON', 'INO_PROCURE_AGENT_1',
    'INO_PROCURE_AGENT_2', 'INO_PROCURE_AGENT_3', 'OTHER_HYPERTENS_MED_DON',
    'BLOOD_INF_CONF_DON', 'HIST_DIABETES_DON', 'URINE_INF_CONF_DON'
]
liverdrop = liverdrop.drop(donorunim, axis=1)

#replace missing value indicators with nan
liverdrop = liverdrop.replace([998], np.nan)

#recode serology level: positive =1, Negative =0, and missing or unknown is missing
liverdrop = liverdrop.replace('P', 1)
liverdrop = liverdrop.replace('N', 0)
liverdrop = liverdrop.replace(['ND', 'U', 'I', 'PD', 'C'], np.nan)

#dropping as new variable cmv_status has very little missing and the difference is likely not
#important knowing igg versus igm
liverdrop = liverdrop.drop('CMV_IGG', axis=1)
liverdrop = liverdrop.drop('CMV_IGM', axis=1)

#Diagnosis codes have many levels. For variables dng_tcr, dgn2_tcr, and diag
#I create 17 levels. We can collapse some of the less frequent levels into the other category
#if necessary.
ahn = [4100, 4101, 4102, 4103, 4104, 4105, 4106, 4107, 4108, 4110, 4217]
auto = [4212]
crypto = [4213, 4208]
etoh = [4215]
etohhcv = [4216]
hbv = [4202, 4592]
hcc = [4400, 4401, 4402]
hcv = [4204, 4593]
nash = [4214]
pbc = [4220]
psc = [4240, 4241, 4242, 4245]
alpha = [4300]
failure = [4598]
cholangio = [4403]
iron = [4302]
wilson = [4301]
poly = [4451]

#variable DIAG refers to primary diagnosis at time of transplant
liverdrop['DIAG'] = liverdrop.DIAG.replace(ahn, 1)
liverdrop['DIAG'] = liverdrop.DIAG.replace(hcc, 2)
liverdrop['DIAG'] = liverdrop.DIAG.replace(auto, 3)
liverdrop['DIAG'] = liverdrop.DIAG.replace(crypto, 4)
liverdrop['DIAG'] = liverdrop.DIAG.replace(etoh, 5)
liverdrop['DIAG'] = liverdrop.DIAG.replace(etohhcv, 6)
liverdrop['DIAG'] = liverdrop.DIAG.replace(hbv, 7)
liverdrop['DIAG'] = liverdrop.DIAG.replace(hcv, 8)
liverdrop['DIAG'] = liverdrop.DIAG.replace(nash, 9)
liverdrop['DIAG'] = liverdrop.DIAG.replace(pbc, 10)
liverdrop['DIAG'] = liverdrop.DIAG.replace(psc, 11)
liverdrop['DIAG'] = liverdrop.DIAG.replace(alpha, 12)
liverdrop['DIAG'] = liverdrop.DIAG.replace(failure, 13)
liverdrop['DIAG'] = liverdrop.DIAG.replace(cholangio, 14)
liverdrop['DIAG'] = liverdrop.DIAG.replace(iron, 15)
liverdrop['DIAG'] = liverdrop.DIAG.replace(wilson, 16)
liverdrop['DIAG'] = liverdrop.DIAG.replace(poly, 17)

#create variable named diag1 where the variable will be 0 to indicate other diagnosis

liverdrop['diag1'] = 0
liverdrop['diag1'] = np.where(liverdrop.DIAG == 1, 1, liverdrop.diag1)
liverdrop['diag1'] = np.where(liverdrop.DIAG == 2, 2, liverdrop.diag1)
liverdrop['diag1'] = np.where(liverdrop.DIAG == 3, 3, liverdrop.diag1)
liverdrop['diag1'] = np.where(liverdrop.DIAG == 4, 4, liverdrop.diag1)
liverdrop['diag1'] = np.where(liverdrop.DIAG == 5, 5, liverdrop.diag1)
liverdrop['diag1'] = np.where(liverdrop.DIAG == 6, 6, liverdrop.diag1)
liverdrop['diag1'] = np.where(liverdrop.DIAG == 7, 7, liverdrop.diag1)
liverdrop['diag1'] = np.where(liverdrop.DIAG == 8, 8, liverdrop.diag1)
liverdrop['diag1'] = np.where(liverdrop.DIAG == 9, 9, liverdrop.diag1)
liverdrop['diag1'] = np.where(liverdrop.DIAG == 10, 10, liverdrop.diag1)
liverdrop['diag1'] = np.where(liverdrop.DIAG == 11, 11, liverdrop.diag1)
liverdrop['diag1'] = np.where(liverdrop.DIAG == 12, 12, liverdrop.diag1)
liverdrop['diag1'] = np.where(liverdrop.DIAG == 13, 13, liverdrop.diag1)
liverdrop['diag1'] = np.where(liverdrop.DIAG == 14, 14, liverdrop.diag1)
liverdrop['diag1'] = np.where(liverdrop.DIAG == 15, 15, liverdrop.diag1)
liverdrop['diag1'] = np.where(liverdrop.DIAG == 16, 16, liverdrop.diag1)
liverdrop['diag1'] = np.where(liverdrop.DIAG == 17, 17, liverdrop.diag1)
liverdrop['diag1'] = np.where(np.isnan(liverdrop.DIAG), np.nan,
                              liverdrop.diag1)
liverdrop['diag1'] = np.where(liverdrop.DIAG.isin([999, np.nan]), 18,
                              liverdrop.diag1)

#variable dnn_tcr refers to primary diagnosis at the time of listing
#variable dgn2_tcr refers to secondary diagnosis at the time of listing
#I apply the same recoding as above

liverdrop['DGN_TCR'] = liverdrop.DGN_TCR.replace(ahn, 1)
liverdrop['DGN_TCR'] = liverdrop.DGN_TCR.replace(hcc, 2)
liverdrop['DGN_TCR'] = liverdrop.DGN_TCR.replace(auto, 3)
liverdrop['DGN_TCR'] = liverdrop.DGN_TCR.replace(crypto, 4)
liverdrop['DGN_TCR'] = liverdrop.DGN_TCR.replace(etoh, 5)
liverdrop['DGN_TCR'] = liverdrop.DGN_TCR.replace(etohhcv, 6)
liverdrop['DGN_TCR'] = liverdrop.DGN_TCR.replace(hbv, 7)
liverdrop['DGN_TCR'] = liverdrop.DGN_TCR.replace(hcv, 8)
liverdrop['DGN_TCR'] = liverdrop.DGN_TCR.replace(nash, 9)
liverdrop['DGN_TCR'] = liverdrop.DGN_TCR.replace(pbc, 10)
liverdrop['DGN_TCR'] = liverdrop.DGN_TCR.replace(psc, 11)
liverdrop['DGN_TCR'] = liverdrop.DGN_TCR.replace(alpha, 12)
liverdrop['DGN_TCR'] = liverdrop.DGN_TCR.replace(failure, 13)
liverdrop['DGN_TCR'] = liverdrop.DGN_TCR.replace(cholangio, 14)
liverdrop['DGN_TCR'] = liverdrop.DGN_TCR.replace(iron, 15)
liverdrop['DGN_TCR'] = liverdrop.DGN_TCR.replace(wilson, 16)
liverdrop['DGN_TCR'] = liverdrop.DGN_TCR.replace(poly, 17)

liverdrop['DGN_TCR1'] = 0
liverdrop['DGN_TCR1'] = np.where(liverdrop.DGN_TCR == 1, 1, liverdrop.DGN_TCR1)
liverdrop['DGN_TCR1'] = np.where(liverdrop.DGN_TCR == 2, 2, liverdrop.DGN_TCR1)
liverdrop['DGN_TCR1'] = np.where(liverdrop.DGN_TCR == 3, 3, liverdrop.DGN_TCR1)
liverdrop['DGN_TCR1'] = np.where(liverdrop.DGN_TCR == 4, 4, liverdrop.DGN_TCR1)
liverdrop['DGN_TCR1'] = np.where(liverdrop.DGN_TCR == 5, 5, liverdrop.DGN_TCR1)
liverdrop['DGN_TCR1'] = np.where(liverdrop.DGN_TCR == 6, 6, liverdrop.DGN_TCR1)
liverdrop['DGN_TCR1'] = np.where(liverdrop.DGN_TCR == 7, 7, liverdrop.DGN_TCR1)
liverdrop['DGN_TCR1'] = np.where(liverdrop.DGN_TCR == 8, 8, liverdrop.DGN_TCR1)
liverdrop['DGN_TCR1'] = np.where(liverdrop.DGN_TCR == 9, 9, liverdrop.DGN_TCR1)
liverdrop['DGN_TCR1'] = np.where(liverdrop.DGN_TCR == 10, 10,
                                 liverdrop.DGN_TCR1)
liverdrop['DGN_TCR1'] = np.where(liverdrop.DGN_TCR == 11, 11,
                                 liverdrop.DGN_TCR1)
liverdrop['DGN_TCR1'] = np.where(liverdrop.DGN_TCR == 12, 12,
                                 liverdrop.DGN_TCR1)
liverdrop['DGN_TCR1'] = np.where(liverdrop.DGN_TCR == 13, 13,
                                 liverdrop.DGN_TCR1)
liverdrop['DGN_TCR1'] = np.where(liverdrop.DGN_TCR == 14, 14,
                                 liverdrop.DGN_TCR1)
liverdrop['DGN_TCR1'] = np.where(liverdrop.DGN_TCR == 15, 15,
                                 liverdrop.DGN_TCR1)
liverdrop['DGN_TCR1'] = np.where(liverdrop.DGN_TCR == 16, 16,
                                 liverdrop.DGN_TCR1)
liverdrop['DGN_TCR1'] = np.where(liverdrop.DGN_TCR == 17, 17,
                                 liverdrop.DGN_TCR1)
liverdrop['DGN_TCR1'] = np.where(np.isnan(liverdrop.DGN_TCR), np.nan,
                                 liverdrop.DGN_TCR1)
liverdrop['DGN_TCR1'] = np.where(liverdrop.DGN_TCR.isin([999]), 18,
                                 liverdrop.DGN_TCR1)

liverdrop['DGN2_TCR'] = liverdrop.DGN2_TCR.replace(ahn, 1)
liverdrop['DGN2_TCR'] = liverdrop.DGN2_TCR.replace(hcc, 2)
liverdrop['DGN2_TCR'] = liverdrop.DGN2_TCR.replace(auto, 3)
liverdrop['DGN2_TCR'] = liverdrop.DGN2_TCR.replace(crypto, 4)
liverdrop['DGN2_TCR'] = liverdrop.DGN2_TCR.replace(etoh, 5)
liverdrop['DGN2_TCR'] = liverdrop.DGN2_TCR.replace(etohhcv, 6)
liverdrop['DGN2_TCR'] = liverdrop.DGN2_TCR.replace(hbv, 7)
liverdrop['DGN2_TCR'] = liverdrop.DGN2_TCR.replace(hcv, 8)
liverdrop['DGN2_TCR'] = liverdrop.DGN2_TCR.replace(nash, 9)
liverdrop['DGN2_TCR'] = liverdrop.DGN2_TCR.replace(pbc, 10)
liverdrop['DGN2_TCR'] = liverdrop.DGN2_TCR.replace(psc, 11)
liverdrop['DGN2_TCR'] = liverdrop.DGN2_TCR.replace(alpha, 12)
liverdrop['DGN2_TCR'] = liverdrop.DGN2_TCR.replace(failure, 13)
liverdrop['DGN2_TCR'] = liverdrop.DGN2_TCR.replace(cholangio, 14)
liverdrop['DGN2_TCR'] = liverdrop.DGN2_TCR.replace(iron, 15)
liverdrop['DGN2_TCR'] = liverdrop.DGN2_TCR.replace(wilson, 16)
liverdrop['DGN2_TCR'] = liverdrop.DGN2_TCR.replace(poly, 17)

liverdrop['DGN2_TCR2'] = 0
liverdrop['DGN2_TCR2'] = np.where(liverdrop.DGN2_TCR == 1, 1,
                                  liverdrop.DGN2_TCR2)
liverdrop['DGN2_TCR2'] = np.where(liverdrop.DGN2_TCR == 2, 2,
                                  liverdrop.DGN2_TCR2)
liverdrop['DGN2_TCR2'] = np.where(liverdrop.DGN2_TCR == 3, 3,
                                  liverdrop.DGN2_TCR2)
liverdrop['DGN2_TCR2'] = np.where(liverdrop.DGN2_TCR == 4, 4,
                                  liverdrop.DGN2_TCR2)
liverdrop['DGN2_TCR2'] = np.where(liverdrop.DGN2_TCR == 5, 5,
                                  liverdrop.DGN2_TCR2)
liverdrop['DGN2_TCR2'] = np.where(liverdrop.DGN2_TCR == 6, 6,
                                  liverdrop.DGN2_TCR2)
liverdrop['DGN2_TCR2'] = np.where(liverdrop.DGN2_TCR == 7, 7,
                                  liverdrop.DGN2_TCR2)
liverdrop['DGN2_TCR2'] = np.where(liverdrop.DGN2_TCR == 8, 8,
                                  liverdrop.DGN2_TCR2)
liverdrop['DGN2_TCR2'] = np.where(liverdrop.DGN2_TCR == 9, 9,
                                  liverdrop.DGN2_TCR2)
liverdrop['DGN2_TCR2'] = np.where(liverdrop.DGN2_TCR == 10, 10,
                                  liverdrop.DGN2_TCR2)
liverdrop['DGN2_TCR2'] = np.where(liverdrop.DGN2_TCR == 11, 11,
                                  liverdrop.DGN2_TCR2)
liverdrop['DGN2_TCR2'] = np.where(liverdrop.DGN2_TCR == 12, 12,
                                  liverdrop.DGN2_TCR2)
liverdrop['DGN2_TCR2'] = np.where(liverdrop.DGN2_TCR == 13, 13,
                                  liverdrop.DGN2_TCR2)
liverdrop['DGN2_TCR2'] = np.where(liverdrop.DGN2_TCR == 14, 14,
                                  liverdrop.DGN2_TCR2)
liverdrop['DGN2_TCR2'] = np.where(liverdrop.DGN2_TCR == 15, 15,
                                  liverdrop.DGN2_TCR2)
liverdrop['DGN2_TCR2'] = np.where(liverdrop.DGN2_TCR == 16, 16,
                                  liverdrop.DGN2_TCR2)
liverdrop['DGN2_TCR2'] = np.where(liverdrop.DGN2_TCR == 17, 17,
                                  liverdrop.DGN2_TCR2)
liverdrop['DGN2_TCR2'] = np.where(np.isnan(liverdrop.DGN2_TCR), np.nan,
                                  liverdrop.DGN2_TCR2)
liverdrop['DGN2_TCR2'] = np.where(liverdrop.DGN2_TCR.isin([999]), 18,
                                  liverdrop.DGN2_TCR2)

#remove variables after transformation
liverdrop = liverdrop.drop('DGN_TCR', axis=1)
liverdrop = liverdrop.drop('DIAG', axis=1)
liverdrop = liverdrop.drop('DGN2_TCR', axis=1)

#Can consider creating a variable indicating whether dgn2_tcr2 or dgn_tcr contains the
#diagnosis.
#liverdrop['ahn']=0
#liverdrop['ahn']=np.where(((liverdrop.DGN_TCR1==1) | (liverdrop.DGN2_TCR2==1)) ,1,liverdrop.ahn)

#donor location
#There is one foregin donor, change this to category 5 so that national and foreign are one category
liverdrop['SHARE_TY'] = liverdrop.SHARE_TY.replace(6, 5)

#payment type
#multiple levels, for now created 3, private insurance, public, medicaid

liverdrop['PRI_PAYMENT_TRR1'] = 0
liverdrop['PRI_PAYMENT_TRR1'] = np.where(liverdrop.PRI_PAYMENT_TRR == 1, 1,
                                         liverdrop.PRI_PAYMENT_TRR1)
liverdrop['PRI_PAYMENT_TRR1'] = np.where(liverdrop.PRI_PAYMENT_TRR == 2, 2,
                                         liverdrop.PRI_PAYMENT_TRR1)
liverdrop['PRI_PAYMENT_TRR1'] = np.where(np.isnan(liverdrop.PRI_PAYMENT_TRR),
                                         np.nan, liverdrop.PRI_PAYMENT_TRR1)

liverdrop['PRI_PAYMENT_TCR1'] = 0
liverdrop['PRI_PAYMENT_TCR1'] = np.where(liverdrop.PRI_PAYMENT_TCR == 1, 1,
                                         liverdrop.PRI_PAYMENT_TCR1)
liverdrop['PRI_PAYMENT_TCR1'] = np.where(liverdrop.PRI_PAYMENT_TCR == 2, 2,
                                         liverdrop.PRI_PAYMENT_TCR1)
liverdrop['PRI_PAYMENT_TCR1'] = np.where(np.isnan(liverdrop.PRI_PAYMENT_TCR),
                                         np.nan, liverdrop.PRI_PAYMENT_TCR1)

liverdrop = liverdrop.drop('PRI_PAYMENT_TRR', axis=1)
liverdrop = liverdrop.drop('PRI_PAYMENT_TCR', axis=1)

#ethnicity variables
#Recoding American Indian/Alaska Native, Native Hawaiian/Pacific Islander, or multiracial
#to category 6
liverdrop['ETHCAT'] = liverdrop.ETHCAT.replace([6, 9, 7], 6)
liverdrop['ETHCAT_DON'] = liverdrop.ETHCAT_DON.astype('float')
liverdrop['ETHCAT_DON'] = liverdrop.ETHCAT_DON.replace([6, 9, 7], 6)

#Meld exception points

#hcc=[1,3,10]
#amyloid=[2]
#hepatopulm=[5]
#metabolic=[6,7,12]
#hat=[11]
#other=[9]

#we don't know what 14 and 16 are

liverdrop['EXC_DIAG_ID'] = liverdrop.EXC_DIAG_ID.replace([3, 10], 1)
liverdrop['EXC_DIAG_ID'] = liverdrop.EXC_DIAG_ID.replace([6, 7], 12)
liverdrop['EXC_DIAG_ID'] = liverdrop.EXC_DIAG_ID.replace([6, 7], 12)

#Can consider dropping EXC_EVER which indicates whether an exception was ever approved
#this encodes identical data to exc_diag_id which says the type of exception

liverdrop = liverdrop.drop('EXC_EVER', axis=1)

#Functional status
#“func_stat_tcr” and “func_stat_trr” indicate the recipient’s functional status at registration
#and transplantation, respectively. Rather than creating a large number of new
#binary categorical features to represent each of the functional status codes,
#unique codes were collapsed into 3 new categorical features representing low, medium, and high
#levels of functional status.

liverdrop['FUNC_STAT_TCR'] = liverdrop.FUNC_STAT_TCR.replace(
    [1, 2080, 2090, 2100, 4080, 4090, 4100], 1)
liverdrop['FUNC_STAT_TCR'] = liverdrop.FUNC_STAT_TCR.replace(
    [2, 2040, 2050, 2060, 2070, 4040, 4050, 4060, 4070], 2)
liverdrop['FUNC_STAT_TCR'] = liverdrop.FUNC_STAT_TCR.replace(
    [3, 2010, 2020, 2030, 4010, 4020, 4030], 3)

liverdrop['FUNC_STAT_TRR'] = liverdrop.FUNC_STAT_TRR.replace(
    [1, 2080, 2090, 2100, 4080, 4090, 4100], 1)
liverdrop['FUNC_STAT_TRR'] = liverdrop.FUNC_STAT_TRR.replace(
    [2, 2040, 2050, 2060, 2070, 4040, 4050, 4060, 4070], 2)
liverdrop['FUNC_STAT_TRR'] = liverdrop.FUNC_STAT_TRR.replace(
    [3, 2010, 2020, 2030, 4010, 4020, 4030], 3)

#status 1 based on meld_diff_reason_cd
liverdrop['status1'] = 0
liverdrop['status1'] = np.where(liverdrop.MELD_DIFF_REASON_CD.isin([15, 16]),
                                1, liverdrop.status1)

#hcc based on meld_diff_reason_cd
#hcc is in many different variables, slightly different. not sure if we should keep all

liverdrop['statushcc'] = 0
liverdrop['statushcc'] = np.where(liverdrop.MELD_DIFF_REASON_CD.isin([8]), 1,
                                  liverdrop.statushcc)
liverdrop = liverdrop.drop('MELD_DIFF_REASON_CD', axis=1)

#coronary artery disease in donor
liverdrop['coronary'] = 0
liverdrop['coronary'] = np.where((liverdrop.CORONARY_ANGIO_DON == "Y") &
                                 (liverdrop.CORONARY_ANGIO_NORM_DON == 0), 1,
                                 liverdrop.coronary)

liverdrop['coronary'] = np.where((liverdrop.CORONARY_ANGIO_DON == "Y") &
                                 (liverdrop.CORONARY_ANGIO_NORM_DON == "Y"), 2,
                                 liverdrop.coronary)

liverdrop = liverdrop.drop('CORONARY_ANGIO_DON', axis=1)
liverdrop = liverdrop.drop('CORONARY_ANGIO_NORM_DON', axis=1)

#Liver Biopsy results
#fat percent of 30% often clinically used. right now used this as cutoff, but can consider
#treating it as continuous. How could we handle this if only some donors had biopsies

liverdrop['macro'] = 0
liverdrop['macro'] = np.where(
    (liverdrop.LI_BIOPSY == "Y") & (liverdrop.MACRO_FAT_LI_DON < 30), 1,
    liverdrop.macro)

liverdrop['macro'] = np.where(
    (liverdrop.LI_BIOPSY == "Y") & (liverdrop.MACRO_FAT_LI_DON >= 30), 2,
    liverdrop.macro)

liverdrop['micro'] = 0
liverdrop['micro'] = np.where(
    (liverdrop.LI_BIOPSY == "Y") & (liverdrop.MICRO_FAT_LI_DON < 30), 1,
    liverdrop.micro)

liverdrop['micro'] = np.where(
    (liverdrop.LI_BIOPSY == "Y") & (liverdrop.MICRO_FAT_LI_DON >= 30), 2,
    liverdrop.micro)

liverdrop = liverdrop.drop('LI_BIOPSY', axis=1)
liverdrop = liverdrop.drop('MICRO_FAT_LI_DON', axis=1)
liverdrop = liverdrop.drop('MACRO_FAT_LI_DON', axis=1)

#Citizenship
liverdrop['CITIZENSHIP'] = liverdrop.CITIZENSHIP.replace([2, 3, 4, 5, 6], 0)

liverdrop['CITIZENSHIP_DON'] = liverdrop.CITIZENSHIP_DON.replace(
    [2, 3, 4, 5, 6], 0)

#creatining binary for diabetes in recipient yes versus no
liverdrop['diabbin'] = liverdrop.DIAB
liverdrop['diabbin'] = liverdrop.diabbin.replace([2, 3, 4, 5], 0)
liverdrop = liverdrop.drop('DIAB', axis=1)

#creating binary for donor insulin dep diabetes. making assumption missing is no
liverdrop['HIST_INSULIN_DEP_DON'] = liverdrop.HIST_INSULIN_DEP_DON.replace(
    [np.nan], 0)

#creating binary for cod of donor as natural cause versus other
liverdrop['deathcirc'] = 0

liverdrop['deathcirc'] = np.where(liverdrop.DEATH_CIRCUM_DON == 6, 1,
                                  liverdrop.deathcirc)
liverdrop = liverdrop.drop('DEATH_CIRCUM_DON', axis=1)

#grouping some causes of death together. not really sure if there is any strong reason to
#group like this
liverdrop['death_mech_don_group'] = liverdrop['DEATH_MECH_DON']
liverdrop['death_mech_don_group'] = liverdrop.death_mech_don_group.replace(
    [2, 4, 6, 9, 10], 1)
liverdrop['death_mech_don_group'] = liverdrop.death_mech_don_group.replace(
    [8, 9], 7)
liverdrop = liverdrop.drop('DEATH_MECH_DON', axis=1)

#status 1 based on end_stat and init_stat
liverdrop['meldstat'] = 0
liverdrop['meldstat'] = np.where(liverdrop.END_STAT.isin([6011, 6010, 6012]),
                                 1, liverdrop.meldstat)

liverdrop['meldstatinit'] = 0
liverdrop['meldstatinit'] = np.where(
    liverdrop.INIT_STAT.isin([6011, 6010, 6012]), 1, liverdrop.meldstat)
liverdrop = liverdrop.drop('END_STAT', axis=1)
liverdrop = liverdrop.drop('INIT_STAT', axis=1)

#split versus whole. can consider keeping more levels
#likely need to change this , look at variable txliv
#liverdrop['split']=0
#liverdrop['split']=np.where(liverdrop.LITYP.isin([20]),1,liverdrop.split)

liverdrop = liverdrop.drop('LITYP', axis=1)

#Insulin dependent diabetes (1 if yes, 0 if no)
liverdrop['insdiab'] = 1
liverdrop['insdiab'] = np.where(liverdrop.INSULIN_DEP_DON.isin([1, np.nan]), 0,
                                liverdrop.insdiab)
liverdrop = liverdrop.drop('INSULIN_DEP_DON', axis=1)

#malig type has tons of categories and I can consider creating a few in the future
#most common cancers are hcc and when I pull out these values I get slightly less than other variables
#for now I will simply create a variable whether the person ever had cancer
#liverdrop['maligtypehcc']=0
#liverdrop['maligtypehcc']=np.where(liverdrop.MALIG_TYPE.isin([4096,8192, 2048]),1,liverdrop.maligtypehcc)

liverdrop = liverdrop.drop('MALIG_TY_TRR', axis=1)
liverdrop = liverdrop.drop('PREV_MALIG_TY', axis=1)

liverdrop['cancer'] = 1
liverdrop['cancer'] = np.where(liverdrop.MALIG_TYPE.isin([np.nan]), 0,
                               liverdrop.cancer)
liverdrop = liverdrop.drop('MALIG_TYPE', axis=1)

#new variables that have lotf of missing data

liverdrop = liverdrop.drop('HBV_NAT', axis=1)
liverdrop = liverdrop.drop('HCV_NAT', axis=1)
liverdrop = liverdrop.drop('HIV_NAT', axis=1)
liverdrop = liverdrop.drop('NEOADJUVANT_THERAPY_TCR', axis=1)
liverdrop = liverdrop.drop('HBV_NAT_DON', axis=1)
liverdrop = liverdrop.drop('HCV_NAT_DON', axis=1)
liverdrop = liverdrop.drop('HIV_NAT_DON', axis=1)
liverdrop = liverdrop.drop('HBV_SURF_TOTAL', axis=1)

#hcc_diag and #hcc_tcr are new variables since 2015. These data I can mostly get from other variables.
#In future I can create a new variable that will be positive if hcc_diag is positive or if diag is positive for hcc
#Also create another variable if dgn2_tcr or dgn_tcr or hcc_tcr suggests hcc

#blood types
liverdrop['abo'] = 0
liverdrop['abo'] = np.where(liverdrop.ABO.isin(["B"]), 1, liverdrop.abo)
liverdrop['abo'] = np.where(liverdrop.ABO.isin(["O"]), 2, liverdrop.abo)
liverdrop['abo'] = np.where(liverdrop.ABO.isin(["A1B", "A2B", 'AB']), 3,
                            liverdrop.abo)

liverdrop['abodon'] = 0
liverdrop['abodon'] = np.where(liverdrop.ABO_DON.isin(["B"]), 1,
                               liverdrop.abodon)
liverdrop['abodon'] = np.where(liverdrop.ABO_DON.isin(["O"]), 2,
                               liverdrop.abodon)
liverdrop['abodon'] = np.where(liverdrop.ABO_DON.isin(["A1B", "A2B", 'AB']), 3,
                               liverdrop.abodon)

liverdrop = liverdrop.drop('ABO', axis=1)
liverdrop = liverdrop.drop('ABO_DON', axis=1)

#let us create a feature that is different between ADMISSION_TIME and TX_DATE
#We can then delete ADMISSION_TIME and TX_DATE

#let us create a feature that is different between ADMIT_DATE_DON and RECOVERY_DATE_DON
#We can then delete ADMIT_DATE_DON and RECOVERY_DATE_DON

liverdrop['admtime'] = pd.to_datetime(liverdrop['ADMISSION_DATE'])
liverdrop['admtimedon'] = pd.to_datetime(liverdrop['ADMIT_DATE_DON'])
liverdrop['recov'] = pd.to_datetime(liverdrop['RECOVERY_DATE_DON'])
liverdrop['tx_date'] = pd.to_datetime(liverdrop['TX_DATE'])
liverdrop['register_date'] = pd.to_datetime(liverdrop['INIT_DATE'])
liverdrop['waitlist_removal_date'] = pd.to_datetime(liverdrop['END_DATE'])
liverdrop['composite_deathdate'] = pd.to_datetime(
    liverdrop['COMPOSITE_DEATH_DATE'])
liverdrop['waitlist_deathdate'] = pd.to_datetime(liverdrop['DEATH_DATE'])

liverdrop['hosptime'] = liverdrop.tx_date - liverdrop.admtime
liverdrop['dontime'] = liverdrop.recov - liverdrop.admtimedon
liverdrop = liverdrop.drop('admtime', axis=1)
liverdrop = liverdrop.drop('ADMISSION_DATE', axis=1)
liverdrop = liverdrop.drop('TX_DATE', axis=1)
liverdrop = liverdrop.drop('INIT_DATE', axis=1)
liverdrop = liverdrop.drop('COMPOSITE_DEATH_DATE', axis=1)
liverdrop = liverdrop.drop('ADMIT_DATE_DON', axis=1)
liverdrop = liverdrop.drop('RECOVERY_DATE_DON', axis=1)
liverdrop = liverdrop.drop('admtimedon', axis=1)
liverdrop = liverdrop.drop('recov', axis=1)

#replace Y with 1, N with 0 and U with np.nan
#make sure to only run this line down here
liverdrop = liverdrop.replace(["U"], np.nan)
liverdrop = liverdrop.replace(["Y"], 1)
liverdrop = liverdrop.replace(["N"], 0)
liverdrop = liverdrop.replace(["M"], 1)
liverdrop = liverdrop.replace(["F"], 0)
liverdrop = liverdrop.replace(["A"], np.nan)
liverdrop = liverdrop.replace(["S"], 1)
liverdrop = liverdrop.replace(["W"], 1)
liverdrop = liverdrop.replace(["non-HCC"], 0)
liverdrop = liverdrop.replace(["HCC"], 1)

#There are a few variables that are nearly identical and not sure the difference between them
#wgt_tcr and init_wgt are examples. init_wgt has fewer missing so will use this

liverdrop = liverdrop.drop('WGT_KG_TCR', axis=1)

liverdrop = liverdrop.drop('END_BMI_CALC', axis=1)

for feature, init_feature in substitution_pairs:
    # fill latest feature with init values if there was no update
    mask = liverdrop[feature].isna() & (~liverdrop[init_feature].isna())
    liverdrop.loc[mask, feature] = liverdrop.loc[mask, init_feature]
    #liverdrop=liverdrop.drop(labels=init_feature,axis='columns')

cats = [
    'ALCOHOL_HEAVY_DON', 'AMIS', 'ANTIHYPE_DON', 'ARGININE_DON', 'ASCITES_TX',
    'BACT_PERIT_TCR', 'BLOOD_INF_DON', 'BMIS', 'ABO_MAT', 'CARDARREST_NEURO',
    'CDC_RISK_HIV_DON', 'CITIZENSHIP', 'CITIZENSHIP_DON', 'CLIN_INFECT_DON',
    'CMV_DON', 'CMV_STATUS', 'COD_CAD_DON', 'DIAL_TX', 'DRMIS',
    'EBV_IGG_CAD_DON', 'EBV_IGM_CAD_DON', 'EBV_SEROSTATUS', 'ECD_DONOR',
    'EDUCATION', 'ENCEPH_TX', 'ETHCAT', 'ETHCAT_DON', 'ETHNICITY',
    'EVER_APPROVED', 'EXC_CASE', 'EXC_DIAG_ID', 'EXC_HCC', 'FUNC_STAT_TCR',
    'FUNC_STAT_TRR', 'GENDER', 'GENDER_DON', 'HBSAB_DON', 'HBV_CORE',
    'HBV_CORE_DON', 'HBV_SUR_ANTIGEN', 'HCC_DIAG', 'HCC_DIAGNOSIS_TCR',
    'HCC_EVER_APPR', 'HCV_SEROSTATUS', 'HEPARIN_DON', 'HEP_C_ANTI_DON',
    'HISTORY_MI_DON', 'HIST_CANCER_DON', 'HIST_CIG_DON', 'HIST_COCAINE_DON',
    'HIST_HYPERTENS_DON', 'HIST_INSULIN_DEP_DON', 'HIST_OTH_DRUG_DON',
    'HIV_SEROSTATUS', 'HLAMIS', 'INIT_ASCITES', 'INIT_DIALYSIS_PRIOR_WEEK',
    'INIT_ENCEPH', 'INOTROP_SUPPORT_DON', 'INSULIN_DON',
    'INTRACRANIAL_CANCER_DON', 'LIFE_SUP_TCR', 'LIFE_SUP_TRR', 'MALIG_TCR',
    'MALIG_TRR', 'MED_COND_TRR', 'NON_HRT_DON', 'ON_VENT_TRR',
    'PORTAL_VEIN_TCR', 'PORTAL_VEIN_TRR', 'PREV_AB_SURG_TCR',
    'PREV_AB_SURG_TRR', 'PREV_TX', 'PREV_TX_ANY', 'PRI_PAYMENT_TCR1',
    'PRI_PAYMENT_TRR1', 'PROTEIN_URINE', 'PSTATUS', 'PT_DIURETICS_DON',
    'PT_STEROIDS_DON', 'PT_T3_DON', 'PT_T4_DON', 'PULM_INF_DON',
    'RECOV_OUT_US', 'REGION', 'SHARE_TY', 'SKIN_CANCER_DON', 'TATTOOS',
    'TIPSS_TCR', 'TIPSS_TRR', 'TRANSFUS_TERM_DON', 'TXLIV', 'URINE_INF_DON',
    'VASODIL_DON', 'VDRL_DON', 'VENTILATOR_TCR', 'WORK_INCOME_TCR',
    'WORK_INCOME_TRR', 'abo', 'abodon', 'cancer', 'coronary',
    'death_mech_don_group', 'deathcirc', 'diabbin', 'diag1', 'insdiab',
    'macro', 'meldstat', 'meldstatinit', 'micro', 'status1', 'statushcc',
    'DIABETES_DON', 'DGN2_TCR2', 'DGN_TCR1', 'DDAVP_DON'
]

liverdrop[cats] = liverdrop[cats].astype('category')

liverdrop = liverdrop.loc[:, ~liverdrop.columns.str.match('Unnamed')]

######################################
# Save cleaned data to csv files.
######################################
save_dir = './Processed Data/Cleaned Transplant Data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

summary = liverdrop.describe(include='all', datetime_is_numeric=True)
summaryt = summary.transpose()
summary = summaryt.sort_index(axis=0)
summary.to_csv(f'{save_dir}/summary.csv')

liverdrop.to_csv(f'{save_dir}/liver_data_cleaned.csv')
print(liverdrop.tx_date.isna().sum(), len(liverdrop.tx_date))
donor_data = liverdrop[['DONOR_ID'] + ORGAN_FEATURES].dropna()
print(len(donor_data))
patient_data = liverdrop[[
    'WL_ID_CODE', 'tx_date', 'register_date', 'composite_deathdate',
    'waitlist_removal_date', 'waitlist_deathdate'
] + PATIENT_FEATURES + INITIAL_PATIENT_FEATURES]

df = patient_data[PATIENT_FEATURES]
is_NaN = df.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = df[row_has_NaN]
print(len(rows_with_NaN), len(df))

donor_data.to_csv(f'{save_dir}/donor_data_cleaned.csv')
patient_data.to_csv(f'{save_dir}/patient_data_cleaned.csv')
