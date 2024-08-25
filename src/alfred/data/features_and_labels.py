feature_columns = ["Close_diff_MA_7", "Volume_diff_MA_7", "Close_diff_MA_30", "Volume_diff_MA_30",
                        "Close_diff_MA_90", "Volume_diff_MA_90", "Close_diff_MA_180", "Volume_diff_MA_180",
                        "Close", "Volume", "reportedEPS", 'Margin_Gross', 'Margin_Operating',
                        'Margin_Net_Profit', "estimatedEPS", "surprise", "surprisePercentage", "10year",
                        "5year", "3year", "2year", "VIX"]

label_columns = ["price_change_term_7", "price_change_term_30", "price_change_term_120",
                      "price_change_term_240"]