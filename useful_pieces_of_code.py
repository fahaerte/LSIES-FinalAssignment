# useful pieces of code
monday = datetime.strptime('2023.05.01 00:00:00', '%Y.%m.%d %H:%M:%S')
friday = datetime.strptime('2023.05.05 23:59:59', '%Y.%m.%d %H:%M:%S')
saturday = datetime.strptime('2023.05.07 00:00:00', '%Y.%m.%d %H:%M:%S')
sunday = datetime.strptime('2023.05.07 23:59:59', '%Y.%m.%d %H:%M:%S')
woindex = pd.date_range(monday, friday, freq='1min')
weindex = pd.date_range(saturday, sunday, freq='1min')