from os import listdir
from os.path import isfile, join

#This file contains a script to implement a function to be able to read files from subfolders of a given folder

def get_files_from_subfolders(my_path,file_names,folders):
    if(my_path not in folders):
        folders.append(my_path)
        path_content = listdir(my_path)

        for f in path_content:
            if(f != '.DS_Store'):
                new_path = join(my_path,f)

                if(isfile(new_path)):
                    if('.wav' in new_path):
                        new_path = new_path.replace('.wav','') 
                        if(new_path not in file_names):
                            file_names.append(new_path)
                    elif('.mid' in new_path):
                        new_path = new_path.replace('.mid','')
                        if(new_path not in file_names):
                            file_names.append(new_path)
                else:
                    get_files_from_subfolders(new_path,file_names,folders)

    return file_names

#Going through the folders to find the wav and the midi files
files = [] 
folders = []
mypath = './' #the starting path
get_files_from_subfolders(mypath,files,folders)

for f in files:
    print(f)
    #doSomething