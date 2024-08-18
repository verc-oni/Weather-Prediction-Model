from modules import Weather, Predictor

url = f"dataset/weatherHistory.csv"
weather = Weather(file=url)
ext = weather.ext

predictor = Predictor(weather.return_data())
predictor.start()



