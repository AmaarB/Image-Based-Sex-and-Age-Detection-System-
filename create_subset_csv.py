import pandas as pd

path = "list_attr_celeba.txt"

read_file_obj = open(path, "r")# read  attribute file list_attr_celeba.txt
lines = read_file_obj.readlines()                                           # from the given data set

write_file_obj = open('processed_list_attr_celeba.txt', 'w')# write the processed data in  
for line in lines:                                         # processed_list_attr_celeba.txt
    x = (line).replace("  "," ") # the space between data in csv files is replaced with cp,,a 
    x = x.replace(" ",",")
    write_file_obj.writelines(x)

import pandas as pd
df = pd.read_csv("processed_list_attr_celeba.txt")   # random blakc cpulm is created next line deletes it 
df = df.iloc[: , :-1] # drop the last column that were generated due to our processing on text file

subset = df[["202599","Male","Young"]] # get the required cloumns data from the whole dataframe df
subset.replace(-1,0,inplace =True) 

#processed data
print(subset.head())

subset.to_csv("subset.csv",index=False) # save the processed dataframe into a csv file
