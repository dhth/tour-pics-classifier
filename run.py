from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse

from fastai.vision import ImageDataBunch, cnn_learner, open_image, get_transforms, imagenet_stats, models, show_image
from pathlib import Path

from io import BytesIO

import sys
import uvicorn
import aiohttp
import asyncio
from PIL import Image as PILImage
from fastai.vision import image2np
import base64

def encode(img):
    img = (image2np(img.data) * 255).astype('uint8')
    pil_img = PILImage.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


app = Starlette()

path = Path('data/')

classes = ['landscape', 'people-close-up', 'people-landscape']
data2 = ImageDataBunch.single_from_classes(path, classes, size=224).normalize(imagenet_stats)
learn = cnn_learner(data2, models.resnet34, pretrained=False)
learn.load('model')
# learn = load_learner(path, 'model.pth')

index_html = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <meta name="description" content="">
    <meta name="author" content="">

    <title>Tour pic classifier</title>

    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css" integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">

  </head>

  <body>
<div class="container-fluid">
	<div class="row" style="padding-top:10px;text-align: center;">
		<div class="col-md-6 offset-md-3 col-sm-12">
				<h2>
          <a href="/">Tour Pictures Classifier</a>
				</h2>
				<p>
					This is an image classifier API that returns one of the following three predictions:
				</p>
        <div class="col-md-6 offset-md-3 col-sm-12">
          <ul style="list-style-position:inside;text-align: left">
            <li>close up shot of people</li>
             <li>people in foreground, landscape in background</li>
             <li>landscape shot</li>
        </ul>
        </div>
				</p>
		</div>
	</div>

	<div class="row" align="center" style="padding-top:20px;">
		<div class="col-md-8 offset-md-2 col-sm-12">
      <div class="row">
        <div class="col-md-6 offset-md-3 col-sm-12">
          <h2>Examples</h2>
        </div>
        </div>
			<div class="row">
				<div class="col-md-4">
					<img alt="Bootstrap Image Preview" style="width:80%;" src="https://cdn.tourradar.com/s3/review/750x400/136555_2e8b6dfb.jpg" />
          <p>close up shot of people</p>
					<p><a target="_blank_" href="https://cdn.tourradar.com/s3/review/750x400/136555_2e8b6dfb.jpg">image source</a>
					</p>
				</div>
				<div class="col-md-4">
					<img alt="Bootstrap Image Preview" style="width:80%;" src="https://cdn.tourradar.com/s3/review/750x400/133738_4485aa24.jpg" />
          <p>people in foreground, landscape in background</p>
					<p><a target="_blank_" href="https://cdn.tourradar.com/s3/review/750x400/133738_4485aa24.jpg">image source</a>
					</p>
				</div>
<div class="col-md-4">
					<img alt="Bootstrap Image Preview" style="width:80%;" src="https://cdn.tourradar.com/s3/review/750x400/98778_bacc2c2d.jpg" />
  <p>landscape shot</p>
					<p><a target="_blank_" href="https://cdn.tourradar.com/s3/review/750x400/98778_bacc2c2d.jpg">image source</a>
					</p>
				</div>
			</div>
		</div>
	</div>
<hr>
      <div class="row">
        <div class="col-md-4 offset-md-4 col-sm-12">
          <h5>Try it out!</h5>
        </div>
        </div>
	<div class="row">
		<div class="col-md-4 offset-md-4 col-sm-12">
			<form role="form" class="form form-inline" action="/upload" method="post" enctype="multipart/form-data">
				<div class="form-group">
					<input required type="file" class="form-control-file" name="file"/>
				</div>
				<button type="submit" class="btn btn-secondary">
					Upload image
				</button>
			</form>
		</div>
	</div>
	<div class="row">
		<div class="col-md-4 offset-md-4 col-sm-12">
			<form role="form" class="form" action="/classify-url" method="get">
				<div class="form-group">
					 
					<label for="url">
						or enter a URL
					</label>
					<input required type="url" name="url" class="form-control" id="url" placeholder="eg. https://cdn.tourradar.com/s3/review/750x400/133738_4485aa24.jpg"/>
				</div>
				<button type="submit" class="btn btn-secondary">
					Submit
				</button>
			</form>
		</div>
	</div>

<hr>
	<div class="row" align="center">
		<div class="col-md-2 offset-md-5 col-sm-4">
        <p class="mb-1">Made by <a target="_" href="https://github.com/dhth/">dhruv</a></p>
        <ul class="list-inline">
          <li class="list-inline-item"><a target="_" href="https://github.com/dhth/tour-pics-classifier">source</a></li>
        </ul>
		</div>
	</div>
</div>
</body>
</html>
"""


resp_html = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <meta name="description" content="">
    <meta name="author" content="">

    <title>Tour pic classifier</title>

    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css" integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">

  </head>

  <body>
<div class="container-fluid">
	<div class="row" style="padding-top:10px;text-align: center;">
		<div class="col-md-6 offset-md-3 col-sm-12">
				<h2>
          <a href="/">Tour Pictures Classifier</a>
				</h2>
				<p>
					This is an image classifier API that returns one of the following three predictions:
				</p>
        <div class="col-md-6 offset-md-3 col-sm-12">
          <ul style="list-style-position:inside;text-align: left">
            <li>close up shot of people</li>
             <li>people in foreground, landscape in background</li>
             <li>landscape shot</li>
        </ul>
        </div>
				</p>
		</div>
	</div>


	<div class="row" align="center" style="padding-top:20px;">
		<div class="col-md-4 offset-md-4 col-sm-12">
        <figure class="figure">
        <img style="max-width:500px;" src="data:image/png;base64, {}" class="figure-img img-thumbnail input-image">
        </figure>
      <p class="lead">It appears to be {}.</p>

		</div>
	</div>
<hr>
      <div class="row">
        <div class="col-md-4 offset-md-4 col-sm-12">
          <h5>Try it out!</h5>
        </div>
        </div>
	<div class="row">
		<div class="col-md-4 offset-md-4 col-sm-12">
			<form role="form" class="form form-inline" action="/upload" method="post" enctype="multipart/form-data">
				<div class="form-group">
					<input required type="file" class="form-control-file" name="file"/>
				</div>
				<button type="submit" class="btn btn-secondary">
					Upload image
				</button>
			</form>
		</div>
	</div>
	<div class="row">
		<div class="col-md-4 offset-md-4 col-sm-12">
			<form role="form" class="form" action="/classify-url" method="get">
				<div class="form-group">
					 
					<label for="url">
						or enter a URL
					</label>
					<input required type="url" name="url" class="form-control" id="url" placeholder="eg. https://cdn.tourradar.com/s3/review/750x400/133738_4485aa24.jpg"/>
				</div>
				<button type="submit" class="btn btn-secondary">
					Submit
				</button>
			</form>
		</div>
	</div>

<hr>
	<div class="row" align="center">
		<div class="col-md-2 offset-md-5 col-sm-4">
        <p class="mb-1">Made by <a target="_" href="https://github.com/dhth/">dhruv</a></p>
        <ul class="list-inline">
          <li class="list-inline-item"><a target="_" href="https://github.com/dhth/tour-pics-classifier">source</a></li>
        </ul>
		</div>
	</div>
</div>
</body>
</html>
"""




@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    pred_class,pred_idx,outputs = learn.predict(img)
    img_data = encode(img)
    if str(pred_class) == 'landscape':
      resp_str = 'a landscape shot'
    elif str(pred_class) == 'people-landscape': 
      resp_str = 'a shot where people are in front of a landscape background'
    else:
      resp_str = 'a close up shot of people'
    return HTMLResponse(resp_html.format(img_data, resp_str))
    #return JSONResponse({
    #    "prediction": pred_class
    #})



@app.route("/")
def form(request):
    return HTMLResponse(index_html)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8008)
