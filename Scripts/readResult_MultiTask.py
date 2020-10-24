import pickle





name_path = "/lstm_4_capas_subject/escalado/con_weight_0.67/"
dataset = "result_05_fullWindow"

name_path = "../Results/" + dataset + name_path
name_file = "result"
if dataset == "result_05" or dataset == "result_05_fullWindow" or dataset == "result_05_101" or dataset == "result_11" or dataset == "result_15" or dataset == "result_3":
    number_folds = 10
elif dataset == "result_12" or dataset == "result_7":
    number_folds = 5
else:
    print("DATASET ERROR")
    quit()

# ------------------------
loss = 0
acc = 0
acc_fall = 0
acc_adl = 0
maa = 0
acc_callback = 0
sensitivity = 0
specificity = 0
val_loss = 0
val_acc = 0
val_acc_fall = 0
val_acc_adl = 0
val_maa = 0
val_acc_callback = 0
val_sensitivity = 0
val_specificity = 0
lr = 0
# ------------------------
all_acc = 0
all_acc_fall = 0
all_acc_adl = 0
all_maa = 0
all_sensitivity = 0
all_specificity = 0
all_acc_id = 0
all_val_acc = 0
all_val_acc_fall = 0
all_val_acc_adl = 0
all_val_maa = 0
all_val_sensitivity = 0
all_val_specificity = 0
all_val_acc_id = 0
all_val_f1 = 0
all_val_f1_tf = 0
all_val_f1_tf2 = 0
# ------------------------


for i in range(number_folds):
    with open(name_path + name_file + str(i) + '.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        history = pickle.load(f)

    ''' print(history[0].keys())
    print("-------------------------------------------------------")
    print("Results to fold number " + str(i) + " in last batch:")
    print("loss: " + str(history[0]['loss'][-1]))'''
    loss = loss + history[0]['loss'][-1]
    '''print("acc: " + str(history[0]['acc'][-1]))
    acc = acc + history[0]['acc'][-1]
    print("acc_fall: " + str(history[0]['acc_fall'][-1]))
    acc_fall = acc_fall + history[0]['acc_fall'][-1]
    print("acc_adl: " + str(history[0]['acc_adl'][-1]))
    acc_adl = acc_adl + history[0]['acc_adl'][-1]
    print("maa: " + str(history[0]['maa'][-1]))
    maa = maa + history[0]['maa'][-1]
    print("acc_callback: " + str(history[0]['acc_callback'][-1]))
    acc_callback = acc_callback + history[0]['acc_callback'][-1]
    print("sensitivity: " + str(history[0]['sensitivity'][-1]))
    sensitivity = history[0]['sensitivity'][-1]
    print("specificity: " + str(history[0]['specificity'][-1]))
    specificity = history[0]['specificity'][-1]

    print("val_loss: " + str(history[0]['val_loss'][-1]))
    val_loss = val_loss + history[0]['val_loss'][-1]
    print("val_acc: " + str(history[0]['val_acc'][-1]))
    val_acc = val_acc + history[0]['val_acc'][-1]
    print("val_acc_fall: " + str(history[0]['val_acc_fall'][-1]))
    val_acc_fall = val_acc_fall + history[0]['val_acc_fall'][-1]
    print("val_acc_adl: " + str(history[0]['val_acc_adl'][-1]))
    val_acc_adl = val_acc_adl + history[0]['val_acc_adl'][-1]
    print("val_maa: " + str(history[0]['val_maa'][-1]))
    val_maa = val_maa + history[0]['val_maa'][-1]
    print("val_acc_callback: " + str(history[0]['val_acc_callback'][-1]))
    val_acc_callback = val_acc_callback + history[0]['val_acc_callback'][-1]
    print("val_sensitivity: " + str(history[0]['val_sensitivity'][-1]))
    val_sensitivity = history[0]['val_sensitivity'][-1]
    print("val_specificity: " + str(history[0]['val_specificity'][-1]))
    val_specificity = history[0]['val_specificity'][-1]
    print("lr: " + str(history[0]['lr'][-1]))
    lr = lr + history[0]['lr'][-1]'''
    print("-------------------------------------------------------")
    print("Results to fold number " + str(i) + " with all dataset:")

    with open(name_path + name_file + "_all_" + str(i) + '.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        [score_acc, score_fall, score_adl,
         score_maa, score_sensitivity, score_specificity, score_acc_id,
         val_score_acc, val_score_fall,
         val_score_adl, val_score_maa,
         val_score_sensitivity, val_score_specificity, val_score_acc_id, val_f1, val_f1_tf, val_f1_tf2, number_adl, number_fall] = pickle.load(f)
        #val_f1, val_f1_tf, val_f1_tf2, number_adl, number_fall
        # los que sobran son los tres f1

    print("-------------------------------------------------------")
    print("acc: " + str(score_acc))
    all_acc = all_acc + score_acc
    print("acc_fall: " + str(score_fall))
    all_acc_fall = all_acc_fall + score_fall
    print("acc_adl: " + str(score_adl))
    all_acc_adl = all_acc_adl + score_adl
    print("maa: " + str(score_maa))
    all_maa = all_maa + score_maa
    print("sensitivity: " + str(score_sensitivity))
    all_sensitivity = all_sensitivity + score_sensitivity
    print("specificity: " + str(score_specificity))
    all_specificity = all_specificity + score_specificity
    print("acc_id: " + str(score_acc_id))
    all_acc_id = all_acc_id + score_acc_id

    print("val_acc: " + str(val_score_acc))
    all_val_acc = all_val_acc + val_score_acc
    print("val_acc_fall: " + str(val_score_fall))
    all_val_acc_fall = all_val_acc_fall + val_score_fall
    print("val_acc_adl: " + str(val_score_adl))
    all_val_acc_adl = all_val_acc_adl + val_score_adl
    print("val_maa: " + str(val_score_maa))
    all_val_maa = all_val_maa + val_score_maa
    print("val_sensitivity: " + str(val_score_sensitivity))
    all_val_sensitivity = all_val_sensitivity + val_score_sensitivity
    print("val_specificity: " + str(val_score_specificity))
    all_val_specificity = all_val_specificity + val_score_specificity
    print("val_acc_id:" + str(val_score_acc_id))
    all_val_acc_id = all_val_acc_id + val_score_acc_id
    #print("val_f1: " + str(val_f1))
    #all_val_f1 = all_val_f1 + val_f1
    #print("val_f1_tf: " + str(val_f1_tf))
    #all_val_f1_tf = all_val_f1_tf + val_f1_tf
    #print("val_f1_tf2: " + str(val_f1_tf2))
    #all_val_f1_tf2 = all_val_f1_tf2 + val_f1_tf2
    print("number_adl: " + str(number_adl))
    print("number_fall: " + str(number_fall))
    print("-------------------------------------------------------")

print("-------------------------------------------------------")
print("Results mean in last batch:")
print("loss: " + str(loss/number_folds))
'''print("acc: " + str(acc/number_folds))
print("acc_fall: " + str(acc_fall/number_folds))
print("acc_adl: " + str(acc_adl/number_folds))
print("maa: " + str(maa/number_folds))
print("sensitivity: " + str(sensitivity/number_folds))
print("specificity: " + str(specificity/number_folds))
print("acc_callback: " + str(acc_callback/number_folds))
print("val_loss: " + str(val_loss/number_folds))
print("val_acc: " + str(val_acc/number_folds))
print("val_acc_fall: " + str(val_acc_fall/number_folds))
print("val_acc_adl: " + str(val_acc_adl/number_folds))
print("val_maa: " + str(val_maa/number_folds))
print("val_sensitivity: " + str(val_sensitivity/number_folds))
print("val_specificity: " + str(val_specificity/number_folds))
print("val_acc_callback: " + str(val_acc_callback/number_folds))
print("lr: " + str(lr/number_folds))'''
print("-------------------------------------------------------")
print("Results mean with all dataset:")
print("acc: " + str(all_acc/number_folds))
print("acc_fall: " + str(all_acc_fall/number_folds))
print("acc_adl: " + str(all_acc_adl/number_folds))
print("maa: " + str(all_maa/number_folds))
print("sensitivity: " + str(all_sensitivity/number_folds))
print("specificity: " + str(all_specificity/number_folds))
print("acc_id: " + str(all_acc_id/number_folds))
print("val_acc: " + str(all_val_acc/number_folds))
print("val_acc_fall: " + str(all_val_acc_fall/number_folds))
print("val_acc_adl: " + str(all_val_acc_adl/number_folds))
print("val_maa: " + str(all_val_maa/number_folds))
print("val_sensitivity: " + str(all_val_sensitivity/number_folds))
print("val_specificity: " + str(all_val_specificity/number_folds))
print("val_acc_id: " + str(all_val_acc_id/number_folds))
#print("val_f1: " + str(all_val_f1/number_folds))
#print("val_f1_tf: " + str(all_val_f1_tf/number_folds))
#print("val_f1_tf2: " + str(all_val_f1_tf2/number_folds))
print("-------------------------------------------------------")
quit()
