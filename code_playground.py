import requests


def add_user(url, headers, file_paths):
    files = [('files[]', open(file_path, 'rb')) for file_path in file_paths]
    data = {'voice': 'Julius'}

    try:
        response = requests.post(url, files=files, headers=headers, data=data)

        for _, file_handle in files:
            file_handle.close()

        if response.status_code == 200:
            print("Voice cloned successfully")
            print("Response:", response.json())
        else:
            print("Failed to clone voice")
            print("Response status code:", response.status_code)
            try:
                print("Response:", response.json())
            except requests.exceptions.JSONDecodeError:
                print("Response content is not in JSON format")
                print("Response content:", response.text)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")


url = 'http://127.0.0.1:5000/process'
headers = {'key': 'kivsa_ehad',
           'request-type': 'ADD_USER', 'voice_description': ''}
file_paths = ['output.mp4']
files = [('files[]', open(file_path, 'rb')) for file_path in file_paths]
data = {'voice': 'Julius'}
requests.post(url, files=files, headers=headers, data=data)
