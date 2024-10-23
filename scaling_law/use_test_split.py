import pandas as pd

def split_csv(file_path, split_ratio, output_use, output_test, random_seed=42):
    df = pd.read_csv(file_path)
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    split_index = int(len(df_shuffled) * split_ratio)
    df_part1 = df_shuffled.iloc[:split_index]
    df_part2 = df_shuffled.iloc[split_index:]
    df_part1.to_csv(output_use, index=False)
    df_part2.to_csv(output_test, index=False)



if __name__ == "__main__":
    


    
    data_names = {0: "craft", 1: "gender", 2: "diabetes", 3: "adult"}
    data_name = data_names[3]
    parent_path = "/home/ubuntu/"
    file_path = parent_path + f"/data/{data_name}/{data_name}.csv"
    output_use = parent_path + f"/data/{data_name}/use.csv"
    output_test = parent_path + f"/data/{data_name}/test.csv"
    split_csv(file_path, 0.7, output_use, output_test)


