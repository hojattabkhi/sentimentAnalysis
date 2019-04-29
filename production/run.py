from flask import Flask, render_template, request, json
from sentimentApp.classic.predictor import makeReady, predict

application = Flask('Sentiment')
application.config['SECRET_KEY'] = "random string"
application.jinja_options["extensions"].append('jinja2.ext.do')

model_path = './sentimentApp/classic/models/svm.LinearSVC_classifier_for_predict_sample.pk'
vocabulary_path = "./sentimentApp/classic/vector.pickel"
clf, vectorizer = makeReady(model_path, vocabulary_path)

@application.route("/")
def index():
    return render_template('page.html')

@application.route("/getSentenceSentiment", methods=['POST'])
def getSentenceSentiment():
    search_filter = request.json['search_filter']
    try:
        result = predict(clf, vectorizer, search_filter)
        if result and result['status']:
            return json.dumps(result)
    except Exception as e:
        print(e, search_filter)


def main():
    try:
        application.run(debug=True, host='0.0.0.0', threaded=True, port=80)
    except Exception as e:
        print(e)
        main()


if __name__ == '__main__':
    main()
