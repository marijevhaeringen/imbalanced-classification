
def round_decimals(df, round_dict):
    for d in round_dict.keys():
        df[round_dict[d]] = df[round_dict[d]].round(d)
    return df

