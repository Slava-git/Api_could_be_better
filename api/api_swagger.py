from api.app import app
from api.constants import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, RESULTS_FOLDER, THIS_FOLDER
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from flask_restplus import Api, Resource
from werkzeug.datastructures import FileStorage
import os

from Processing_of_gender_swap import gender_swap
api = Api(app)
upload_parser = api.parser()
upload_parser.add_argument('file',
                           location='files',
                           type=FileStorage)

@api.route('/upload/')
@api.expect(upload_parser)
class File(Resource):
    def post(self):
        args = upload_parser.parse_args()
        file = args.get('file')
        assert allowed_file(file.filename)
        path = os.path.join(THIS_FOLDER, UPLOAD_FOLDER, "image.jpg")
        print(path)
        file.save(path)
        gender_swap()
        return "result was saved at {}".format(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

