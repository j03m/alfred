class Stock_Data():
    def __init__(self, root_path, dataset_name, full_stock_path, size, attr=config.TECHNICAL_INDICATORS_LIST,
                 temporal_feature=config.TEMPORAL_FEATURE, scale=True, prediction_len=[2, 5]):
        # size [seq_len, label_len, pred_len]
        self.scale = scale
        self.attr = attr
        self.temporal_feature = temporal_feature
        self.root_path = root_path
        self.full_stock = full_stock_path
        self.ticker_list = config.use_ticker_dict[dataset_name]
        self.border_dates = config.date_dict[dataset_name]
        self.prediction_len = prediction_len

        self.seq_len = size[0]  # seq_len
        self.type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.pred_type_map = {'label_short_term': 0, 'label_long_term': 1}

        self.__read_data__()

    def __read_data__(self):
        scaler = StandardScaler()
        stock_num = len(self.ticker_list)

        full_stock_dir = os.path.join(self.root_path, self.full_stock)

        df_list = []
        for ticket in self.ticker_list:
            temp_df = pd.read_csv(os.path.join(full_stock_dir, ticket + '.csv'),
                                  usecols=['date', 'open', 'close', 'high', 'low', 'volume', 'dopen', 'dclose', 'dhigh',
                                           'dlow', 'dvolume', 'price'])

            temp_df['date'] = temp_df['date'].apply(lambda x: str(x))
            temp_df['date'] = pd.to_datetime(temp_df['date'])
            temp_df['label_short_term'] = temp_df['close'].pct_change(periods=self.prediction_len[0]).shift(
                periods=(-1 * self.prediction_len[0]))
            temp_df['label_long_term'] = temp_df['close'].pct_change(periods=self.prediction_len[1]).shift(
                periods=(-1 * self.prediction_len[1]))
            temp_df['tic'] = ticket
            df_list.append(temp_df)

        df = pd.concat(df_list, ignore_index=True)
        df = df.sort_values(by=['date', 'tic'])

        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
            use_turbulence=False,
            user_defined_feature=False)

        print("generate technical indicator...")
        df = fe.preprocess_data(df)

        # add covariance matrix as states
        df = df.sort_values(['date', 'tic'], ignore_index=True)
        df.index = df.date.factorize()[0]

        cov_list = []
        return_list = []

        # look back is one year
        print("generate convariate matrix...")
        lookback = 252
        for i in range(lookback, len(df.index.unique())):
            data_lookback = df.loc[i - lookback:i, :]
            price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
            return_lookback = price_lookback.pct_change().dropna()
            return_list.append(return_lookback)

            covs = return_lookback.cov().values
            cov_list.append(covs)

        df_cov = pd.DataFrame({'date': df.date.unique()[lookback:], 'cov_list': cov_list, 'return_list': return_list})
        df = df.merge(df_cov, on='date')
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)

        df['date_str'] = df['date'].apply(lambda x: datetime.datetime.strftime(x, '%Y%m%d'))

        # TODO cov_list is a list in a dataframe? da fu?

        dates = df['date_str'].unique().tolist()
        boarder1_ = dates.index(self.border_dates[0])
        boarder1 = dates.index(self.border_dates[1])

        boarder2_ = dates.index(self.border_dates[2])
        boarder2 = dates.index(self.border_dates[3])

        boarder3_ = dates.index(self.border_dates[4])
        boarder3 = dates.index(self.border_dates[5])

        self.boarder_end = [boarder1, boarder2, boarder3]
        self.boarder_start = [boarder1_, boarder2_, boarder3_]

        df_data = df[self.attr]
        df_data = df_data.replace([np.inf], config.INF)
        df_data = df_data.replace([-np.inf], config.INF * (-1))
        if self.scale:
            data = scaler.fit_transform(df_data.values)
        else:
            data = df_data.values

        # TODO you need to understand this more granularly
        cov_list = np.array(df['cov_list'].values.tolist())  # [stock_num*len, stock_num]
        feature_list = np.array(df[self.temporal_feature].values.tolist())  # [stock_num*len, 10]
        close_list = np.array(df['price'].values.tolist())

        # pdb.set_trace()
        # cov_list is a list of matrices where len(cov_list) is the number of days * stocks and each entry is a list is 88 (num_stocks) by 88 covariance matrix
        # we then reshape this to be days x stocks x stock x stock which is each day, index by each stock and then its relationship to other stocks
        # the idea that we didn't start with a "days" length list might be inefficient, but I haven't investigated. Later we everything but the 1st matrix for each stock with the call to
        # data_cov[:, 0, :, :] so - seems wasteful
        data_cov = cov_list.reshape(-1, stock_num, cov_list.shape[1],
                                    cov_list.shape[2])  # [day, num_stocks, num_stocks, num_stocks]

        data_technical = data.reshape(-1, stock_num, len(self.attr))  # [day, stock_num, technical_len]

        # feature list is stocks * days x feature (10) we reshape that into days, stocks, features
        data_feature = feature_list.reshape(-1, stock_num,
                                            len(self.temporal_feature))  # [day, stock_num, temporal_feature_len=10]
        data_close = close_list.reshape(-1, stock_num)

        label_short_term = np.array(df['label_short_term'].values.tolist()).reshape(-1, stock_num)
        label_long_term = np.array(df['label_long_term'].values.tolist()).reshape(-1, stock_num)

        # the final result of data_all is days x stocks x (88 covariance, 8 technicals, 10 features)
        self.data_all = np.concatenate((data_cov[:, 0, :, :], data_technical, data_feature),
                                       axis=-1)  # [days, num_stocks, cov+technical_len+feature_len]
        self.label_all = np.stack((label_short_term, label_long_term), axis=0)  # [2, days, num_stocks, 1]
        self.dates = np.array(dates)
        self.data_close = data_close

        print("data shape: ", self.data_all.shape)
        print("label shape: ", self.label_all.shape)