def filter_df(df, keywords=None):
    keywords = ['resnet50', 'resnet101', 'resnet152',
                "vgg11", "vgg16", "vgg19",
                "vit_b_16", "vit_b_32", "vit_l_16"]
    # 当没有提供关键词时返回原始 DataFrame
    if not keywords:
        print("No keywords provided, skipping filtering")
        return df

    # 初始化过滤掩码
    mask = False
    for kw in keywords:
        # 使用完全匹配（不区分大小写）
        mask |= df['model_name'].str.lower() == kw.lower()

    print(f"Filtering model_name exactly matching: {keywords}")
    return df[mask]
