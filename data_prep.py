import pandas as pd


def load_and_clean_data():
    # Load main dataset
    df = pd.read_csv("/Users/ymadigital/Downloads/Competition/internet_usage.csv")

    # Melt to long format
    df = df.melt(id_vars=['Country Name', 'Country Code'],
                 var_name='Year',
                 value_name='Internet Usage (%)')

    # Convert year to integer
    df['Year'] = df['Year'].astype(int)

    # Load region data (supplementary dataset example)
    regions = pd.read_csv(
        "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv")
    regions = regions[['alpha-3', 'region']].rename(columns={'alpha-3': 'Country Code'})

    # Merge with main data
    df = pd.merge(df, regions, on='Country Code', how='left')

    return df


if __name__ == "__main__":
    df = load_and_clean_data()
    df.to_csv("processed_data.csv", index=False)
