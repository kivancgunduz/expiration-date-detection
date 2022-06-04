import pytesseract
import pandas   

def detect(img):
    print("Start the OCR...")
    #text = pytesseract.image_to_string(img)
    data = pytesseract.image_to_data(img, output_type='data.frame')
    data_text = data.dropna()
    print("dropping row for a word that is less than 30% and equal to 95% of confidence...")
    mask = (data_text["conf"] > 30) & (data_text["conf"] != 95.0) #arbitrary level of confindence for test purpose
    print("dropping empty text...")
    mask2 = (data_text["text"] != "") | (data_text["text"] != " ") 
    
    data_text = data_text[mask]
    df = data_text[mask2]
    list_string = []
    list_conf = []

    [list_string.append(word) for word in data_text["text"]]
    [list_conf.append(word) for word in data_text["conf"]]

    mean = 0

    for score in list_conf:
        mean = mean + score

    if len(list_conf) != 0:
        mean = mean/len(list_conf)

    string = " ".join(list_string)
    
    print("image processed")
    print(f"text is  : '{string}'")
    print(f"List of confidence  : '{list_conf}'")
    print(f"MEAN of confidence : {round(mean,2)}%")
    print(f"PRINT DATAFRAME : {df}")
    return string

def detect2 (img):
    print("Start the OCR...")
    text = pytesseract.image_to_string(img)
    print("image processed")
    print(f"text is  : '{text}'")
    return text
    pytesseract.im