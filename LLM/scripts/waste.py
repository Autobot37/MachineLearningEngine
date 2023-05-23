import os



os.environ['CFLAGS'] = '-I"C:\\Users\\SHIVA SINGH\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\libaio\\include"'
os.environ['LDFLAGS'] = '-L"C:\\Users\\SHIVA SINGH\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\libaio\\lib"'

os.environ['DS_BUILD_AIO'] = '0'
os.environ["DS_BUILD_OPS"] = "1"
