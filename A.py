import requests

def get_coordinates(address):
    base_url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "MyGeocoder/1.0 (your_email@example.com)"}  # ใช้อีเมลของคุณเอง
    params = {"q": address, "format": "json"}
    
    response = requests.get(base_url, params=params, headers=headers)

    if response.status_code != 200:
        print(f"ข้อผิดพลาด HTTP: {response.status_code}")
        return None
    
    try:
        data = response.json()
        if data:
            location = data[0]
            return location["lat"], location["lon"]
        else:
            print("ไม่พบพิกัดที่ตรงกับที่อยู่")
            return None
    except requests.exceptions.JSONDecodeError:
        print("ไม่สามารถแปลงข้อมูล JSON ได้")
        return None

# ตั้งค่าที่อยู่
address = "222 ถนน งามวงศ์วาน แขวงทุ่งสองห้อง เขตหลักสี่ กรุงเทพมหานคร 10210"

coordinates = get_coordinates(address)
if coordinates:
    print(f"พิกัด GPS: {coordinates[0]}, {coordinates[1]}")
else:
    print("ไม่สามารถค้นหาพิกัด GPS ได้")
