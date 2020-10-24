import pickle





name_path = "/late_fusion_4_capas/escalado/con_weight_0.95__sin_weight/"
dataset = "result_05_fullWindow"

name_path = "../Results/" + dataset + name_path
name_file = "result"
if dataset == "result_05" or dataset == "result_05_fullWindow" or dataset == "result_05_101":
    number_folds = 10
elif dataset == "result_12":
    number_folds = 5
else:
    print("DATASET ERROR")
    quit()


# ------------------------
all_acc_1 = 0
all_acc_2 = 0
all_acc_fusion = 0

all_maa_1 = 0
all_maa_2 = 0
all_maa_fusion = 0

all_sensitivity_1 = 0
all_sensitivity_2 = 0
all_sensitivity_fusion = 0

all_specificity_1 = 0
all_specificity_2 = 0
all_specificity_fusion = 0
# ------------------------


for i in range(number_folds):
    with open(name_path + name_file + str(i) + '_late.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        [acc_1, sensitivity_1, specificity_1, maa_1, acc_2, sensitivity_2, specificity_2, maa_2,
         acc_fusion, sensitivity_fusion, specificity_fusion, maa_fusion] = pickle.load(f)


    print("-------------------------------------------------------")
    print("acc_1: " + str(acc_1))
    all_acc_1 = all_acc_1 + acc_1
    print("sensitivity_1: " + str(sensitivity_1))
    all_sensitivity_1 = all_sensitivity_1 + sensitivity_1
    print("specificity_1: " + str(specificity_1))
    all_specificity_1 = all_specificity_1 + specificity_1
    print("maa_1: " + str(maa_1))
    all_maa_1 = all_maa_1 + maa_1

    print("acc_2: " + str(acc_2))
    all_acc_2 = all_acc_2 + acc_2
    print("sensitivity_1: " + str(sensitivity_2))
    all_sensitivity_2 = all_sensitivity_2 + sensitivity_2
    print("specificity_2: " + str(specificity_2))
    all_specificity_2 = all_specificity_2 + specificity_2
    print("maa_2: " + str(maa_2))
    all_maa_2 = all_maa_2 + maa_2

    print("acc_fusion: " + str(acc_fusion))
    all_acc_fusion = all_acc_fusion + acc_fusion
    print("sensitivity_fusion: " + str(sensitivity_fusion))
    all_sensitivity_fusion = all_sensitivity_fusion + sensitivity_fusion
    print("specificity_fusion: " + str(specificity_fusion))
    all_specificity_fusion = all_specificity_fusion + specificity_fusion
    print("maa_1: " + str(maa_fusion))
    all_maa_fusion = all_maa_fusion + maa_fusion
    print("-------------------------------------------------------")

print("-------------------------------------------------------")
print("Results mean with all dataset:")
print("acc_1: " + str(all_acc_1/number_folds))
print("sensitivity_1: " + str(all_sensitivity_1/number_folds))
print("specificity_1: " + str(all_specificity_1/number_folds))
print("maa_1: " + str(all_maa_1/number_folds))

print("acc_2: " + str(all_acc_2/number_folds))
print("sensitivity_2: " + str(all_sensitivity_2/number_folds))
print("specificity_2: " + str(all_specificity_2/number_folds))
print("maa_2: " + str(all_maa_2/number_folds))

print("acc_fusion: " + str(all_acc_fusion/number_folds))
print("sensitivity_fusion: " + str(all_sensitivity_fusion/number_folds))
print("specificity_fusion: " + str(all_specificity_fusion/number_folds))
print("maa_fusion: " + str(all_maa_fusion/number_folds))
print("-------------------------------------------------------")
quit()
