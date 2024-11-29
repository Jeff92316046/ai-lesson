import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import pandas as pd

geolocator = Nominatim(user_agent="ErEr")
training_df = pd.read_csv("lesson8/california_housing_train.csv")
def get_california_county(latitude, longitude):
    """根據經緯度獲取加州的行政區。

    Args:
        latitude: 緯度。
        longitude: 經度。

    Returns:
        加州的行政區名稱，如果找不到則返回 None。
    """
    try:
        time.sleep(0.1)
        location = geolocator.reverse(f"{latitude}, {longitude}", language='en')
        address = location.raw['address']
        county = address.get('county', None)
        print(county)
        if county:
            # 確保只返回加州的行政區
            if "California" in address.get('state', ''):
                return county
            else:
                return None
        else:
            return None
    except GeocoderTimedOut:
        return get_california_county(latitude, longitude)  # 重新嘗試
    except Exception as e:
        print(f"Error")
        return None
training_df['county'] = training_df.apply(lambda row: get_california_county(row['latitude'], row['longitude']), axis=1)
print(training_df['county'].head())
training_df.to_csv("lesson8/output1.csv")