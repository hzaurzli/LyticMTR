import numpy as np
import argparse

def main(dict1,dict2,k):
  dict3 = dict1
  random_keys = np.random.choice(list(dict1.keys()), k, replace = False)
  for random_key in random_keys:
    random_value = dict1[random_key]
    dict2[random_key] = random_value
    del dict3[random_key]
    
  return dict2,dict3


def main_test(dict1,dict2,k):
  random_keys = np.random.choice(list(dict1.keys()), k, replace = False)
  for random_key in random_keys:
    random_value = dict1[random_key]
    dict2[random_key] = random_value
    
  return dict2


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="random sample")
  parser.add_argument("-i", "--input_file", required=True, type=str, help="input feature table (txt,'\t')")
  parser.add_argument("-or", "--output_train_file", required=True, type=str, help="output training feature table (after sample, txt,'\t')")
  parser.add_argument("-oe", "--output_test_file", required=True, type=str, help="output testing feature table (after sample, txt,'\t')")
  parser.add_argument("-ov", "--output_valid_file", required=True, type=str, help="output validation feature table (after sample, txt,'\t')")
  parser.add_argument("-kr", "--sample_kr", required=True, type=int, help="sample number for training")
  parser.add_argument("-ke", "--sample_ke", required=True, type=int, help="sample number for testing")
  parser.add_argument("-kv", "--sample_kv", required=True, type=int, help="sample number for validation")
  Args = parser.parse_args()
  
  f = open(Args.input_file)
  
  old_train_dict = {}
  new_train_dict = {}
  for i in f:
    key = i.strip().split('\t')[0]
    val = '\t'.join(i.strip().split('\t')[1:])
    old_train_dict[key] = val
  
  new_train_dict,old_test_dict = main(old_train_dict,new_train_dict,Args.sample_kr)
  
  with open(Args.output_train_file,'w') as w:
    for key in new_train_dict:
      line = key + '\t' + new_train_dict[key] + '\n'
      w.write(line)
  w.close()
  
  new_test_dict = {}
  #print(test_key)
  new_test_dict = main_test(old_test_dict,new_test_dict,Args.sample_ke)
  
  with open(Args.output_test_file,'w') as w:
    for key in new_test_dict:
      line = key + '\t' + new_test_dict[key] + '\n'
      w.write(line)
  w.close()
  
  old_valid_dict = new_train_dict
  old_valid_dict.update(new_test_dict)
  new_valid_dict = {}
  new_valid_dict_tmp = {}
  
  for key in old_train_dict:
    if key not in old_valid_dict:
      new_valid_dict_tmp[key] = old_train_dict[key]
      
  new_valid_dict = main_test(new_valid_dict_tmp,new_valid_dict,Args.sample_kv)
  
  with open(Args.output_valid_file,'w') as w:
    for key in new_valid_dict:
      line = key + '\t' + new_valid_dict[key] + '\n'
      w.write(line)
  w.close()
  
  
