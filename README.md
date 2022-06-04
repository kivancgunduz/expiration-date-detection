# expiration-date-detection
An API that detect expiration date from the product package's picture based on Deep Learning Algorithms. The project developed for [Delhaize](https://www.aholddelhaize.com/brands/delhaize/) managed by [BeCode](https://becode.org/).
 
 
### Approach

- [Scientific Approach](https://www.sciencedirect.com/science/article/pii/S0957417422006728?via%3Dihub)
- FCOS
- TesseractOCR

### Technologies

``` 
- Python 3.9.5 
- FastAPI 0.78.0
- opencv 4.5.5
- PyTorch 1.5.0
- PyTesseract 0.3.9
``` 

### Installation

    
   - [X] clone repository
   - [X] `pip install -r requirements.txt`
   - [X] run `python main.py`
  
<img src="https://raw.githubusercontent.com/kivancgunduz/expiration-date-detection/master/assets/delhaize_api_main.png" height='500'>
   


### REST-API Endpoints
- GET 
  - /camera/
  - /my_api/
  - /picture/
  - /options/
- POST
  - /take_picture_camera/
  - /load_picture/

### Project Guideline

- Duration: ` 2 Weeks `
- API Deployment: [Heroku](https://delhaize-api.herokuapp.com/)
- Deadline: ` 03.06.22 `

<img src="https://imageio.forbes.com/i-forbesimg/media/lists/companies/delhaize-group_416x416.jpg?format=jpg&height=416&width=416&fit=bounds" height='200'> <img src="https://becode.org/app/uploads/2020/03/cropped-becode-logo-seal.png" height='200'> 
