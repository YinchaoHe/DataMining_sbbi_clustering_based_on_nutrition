import argparse
import csv
import glob
import json
import os
import shutil
import numpy as np
import pandas as pd


def extract_text_files(country):
    os.system("find . -name original_recipes_info > re_info_path.txt")
    try:
        os.mkdir(country)
    except:
        print('this folder exists')
    with open("re_info_path.txt") as f:
        paths = f.readlines()
    for path in paths:
        output_path = path.replace('\n', '').split('original_recipes_info')[0]
        output_path = output_path.split('./')[1]
        dir = output_path.split('/')[0] + '_text_info'
        dir = country + '/' + dir
        try:
            os.mkdir(dir)
        except:
            print('this folder exists')
        output_path = output_path.replace('/', '_')
        print("This is output_path: " + output_path)
        path = path.replace('\n', '') + '/'
        print("this is grap path: " + path)
        files=glob.glob(path + '*')
        for file in files:
            print(file)
            directory = output_path
            filename = file.split('/')[-1]
            directory += filename
            shutil.move(file, dir + '/' + directory)
            print(file)





def nutrition_json2cvs(country):
    #get all recipes' calories information
    all_recipes_calo = {}
    calo_path = 'world_cuisine_extend_calories'
    if os.path.isdir(calo_path):
        os.chdir(calo_path)
        calo_folders = glob.glob( country + '*')
    else:
        print("Please add a folder called world_cuisine_extend_calories")

    for calo_folder in calo_folders:
        os.chdir(calo_folder)
        calo_files = glob.glob('*')
        for calo_file in calo_files:
            with open(calo_file, 'r') as f:
                recipes_calo = json.load(f)
                for recipe_calo in recipes_calo:
                    calo = []
                    for key in recipe_calo.keys():
                        if key == 'recipe_ID':
                            calo.append(int(recipe_calo[key]))
                        else:
                            calo.append(float(recipe_calo[key]))
                    try:
                        all_recipes_calo[str(calo[0])] = calo[1]
                    except:
                        all_recipes_calo[str(calo[0])] = 1000000000000000

    #get other nutrition informtion
    root_path = os.getcwd()
    root_path = root_path.split('/world_cuisine_extend_calories')[0]
    os.chdir(root_path)
    csv_columns = ['Recipe_ID', 'calories', 'Total Fat', 'Saturated Fat', 'Cholesterol', 'Sodium', 'Potassium','Total Carbohydrates', 'Dietary Fiber', 'Protein', 'Sugars', 'Vitamin A', 'Vitamin C', 'Calcium', 'Iron', 'Thiamin', 'Niacin', 'Vitamin B6', 'Magnesium', 'Folate']
    folders = glob.glob(country + '/*')
    for folder in folders:
        print(folders)
        files = glob.glob(folder + '/*')
        for file in files:
            print(file)
            with open(file, 'r') as f:
                recipes = json.load(f)
            dict_data = []
            for recipe in recipes:
                if (len(recipe['nutrition']) > 0):
                    for key in recipe['nutrition'].keys():
                        if 'mcg' in recipe['nutrition'][key]:
                            recipe['nutrition'][key] = recipe['nutrition'][key].split('mcg')[0]
                            if '<' in recipe['nutrition'][key]:
                                recipe['nutrition'][key] = float(recipe['nutrition'][key].split('<')[-1]) / 2.0

                        elif 'mg' in recipe['nutrition'][key]:
                            recipe['nutrition'][key] = recipe['nutrition'][key].split('mg')[0]
                            if '<' in recipe['nutrition'][key]:
                                recipe['nutrition'][key] = float(recipe['nutrition'][key].split('<')[-1]) / 2.0

                        elif 'g' in recipe['nutrition'][key]:
                            recipe['nutrition'][key] = recipe['nutrition'][key].split('g')[0]
                            if '<' in recipe['nutrition'][key]:
                                recipe['nutrition'][key] = float(recipe['nutrition'][key].split('<')[-1]) / 2.0

                        elif 'IU' in recipe['nutrition'][key]:
                            recipe['nutrition'][key] = recipe['nutrition'][key].split('IU')[0]
                            if '<' in recipe['nutrition'][key]:
                                recipe['nutrition'][key] = float(recipe['nutrition'][key].split('<')[-1]) / 2.0

                    recipe['nutrition']['Recipe_ID'] = recipe['recipe_ID']
                    try:
                        recipe['nutrition']['calories'] = all_recipes_calo[recipe['recipe_ID']]
                    except:
                        recipe['nutrition']['calories'] = 100000000
                    dict_data.append(recipe['nutrition'])

            new_csv_columns =  ['Recipe_ID', 'calories', 'TotalFat', 'SaturatedFat', 'Cholesterol', 'Sodium', 'Potassium','TotalCarbohydrates', 'DietaryFiber', 'Protein', 'Sugars', 'VitaminA', 'VitaminC', 'Calcium', 'Iron', 'Thiamin', 'Niacin', 'VitaminB6', 'Magnesium', 'Folate']

            csv_file =  country + '/' + country + "_recipes_nutrition.csv"
            if os.path.isfile(csv_file):
                try:
                    with open(csv_file, 'a+') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                        for data in dict_data:
                            writer.writerow(data)
                except IOError:
                    print("I/O error")
            else:
                try:
                    with open(csv_file, 'w') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                        writer.writeheader()
                        for data in dict_data:
                            writer.writerow(data)
                    df = pd.read_csv(csv_file, header=0)
                    df.columns = new_csv_columns
                    df.to_csv(csv_file, index=False)
                except IOError:
                    print("I/O error")

            with open( country + '/' +country+'_integration_all_calo.json', 'w') as f:
                json.dump(all_recipes_calo, f)

def combination_csv():
    chosen_data_with_calor = [['calories','TotalFat','SaturatedFat','Cholesterol','Sodium','Potassium','TotalCarbohydrates','Protein','Sugars', 'VitaminA']]

    os.system("find . -name *_recipes_nutrition.csv > re_nutri_path.txt")
    # try:
    #     os.mkdir()
    # except:
    #     print('this folder exists')
    with open("re_nutri_path.txt") as f:
        paths = f.readlines()
    for path in paths:
        csv_file = path.split('\n')[0]
        chosen_nutritions_with_cal = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]
        chosen_nutritions_without_cal = [1, 2, 3, 4, 5, 6, 8, 9, 10]
        nutrition_data1 = np.genfromtxt(csv_file, delimiter=',', dtype=float, skip_header=True)
        nutrition_data1 = np.delete(nutrition_data1, 0, 1)
        chosen_data_with_calor_one_country = nutrition_data1[:, chosen_nutritions_with_cal]
        chosen_data_with_calor = np.vstack((chosen_data_with_calor, chosen_data_with_calor_one_country))
    pd.DataFrame(chosen_data_with_calor).to_csv("all_recipes_nutrients_info.csv", index=False)
    print(1)

def main():
    combination_csv()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-c", "--country", help="country information", required=True)
    # parser.add_argument("-n", "--nutrition", help="nutrition information", default=False)
    # country = parser.parse_args().country
    # isNutrition = parser.parse_args().nutrition
    # if isNutrition == False:
    #     extract_text_files(country)
    # else:
    #     nutrition_json2cvs(country)

if __name__ == '__main__':
    main()