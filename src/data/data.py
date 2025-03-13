import re
import unicodedata

import polars as pl

def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    # (57) や任意の数字の括弧を削除
    text = re.sub(r"\(\d+\)", "", text)
    # 【要約】を削除
    text = text.replace("【要約】", "")
    text = text.replace("【課題】", "")
    text = text.replace("【解決手段】", "")
    text = text.replace("(修正有)", "")
    # 【選択図】図３（数字は任意）を削除
    text = re.sub(r"【選択図】図\d+", "", text)
    text = text.replace("\n", "")
    return text

def get_fi_list(elements):
    elements = elements.split(",")
    elements = list(set(re.match(r"([^/]+)", element).group(1) for element in elements if "/" in element))
    elements = ", ".join(elements)
    return elements

def extract_dataframe(df):
    df = df.select([
        pl.col("要約").map_elements(clean_text, return_dtype=pl.String).alias("summary"),
        pl.col("FI").map_elements(get_fi_list, return_dtype=pl.String).alias("FI")
    ])
    return df

def main():
    dataframes = []
    for year in ["2018", "2019", "2020", "2021", "2022", "2023", "2024"]:
        df = pl.read_csv(f"./data/raw/patent_deeplearning_{year}.csv")
        df = extract_dataframe(df)
        if year == "2024":
            df.write_csv("./data/processed/test.csv")
        dataframes.append(df)
        
    df_train = pl.concat(dataframes)
    df_train.write_csv("./data/processed/train.csv")

if __name__ == "__main__":
    main()
    