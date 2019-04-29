# Sentiment Analysis in Persian
use machine learning techniques in order to classify persian sentences into Positive/Negative polarities

## Techniques
1. Classic
* svm.LinearSVC  Classifier
```bash
Mean of  10 fold precision:	 0.927669950643
Mean of  10 fold recall:	 0.978142752915
Mean of  10 fold F1_score:	 0.952224460766
Mean of  10 accuracy:	 0.915778413179
```
* Naive_bayes  Classifier
```bash
Mean of  10 fold precision:	 0.858574907042
Mean of  10 fold recall:	 0.999770642202
Mean of  10 fold F1_score:	 0.923772145124
Mean of  10 accuracy:	 0.858439622871
```
* KNeighbors  Classifier
```bash
Mean of  10 fold precision:	 0.88121139396
Mean of  10 fold recall:	 0.996167150367
Mean of  10 fold F1_score:	 0.935142050705
Mean of  10 accuracy:	 0.881452919575
```
2. Neural Networks
* NOT_IMPLEMENTED_YET

for `detailed report` read reported files in `development/classic`
## DataSet
scraped `http://digikala.com`(the biggest e-commerce startup in Iran) and genarate a product reviews dataSet consist of total 51200 reviews and we select just 8000 positive and 8000 negative reviews and then clean them by hand

### Sample dataSet
| Score        | Text                                                                                        |
| ------------- |:------------------------------------------------------------------------------------------:|
| 5      | من این را از شگفت انگیز خریدم و ازش راضی ام خیلی خوبه فقط مشکل اش صداش زیاد هست ولی نمیشه گفت مشکل . |
| 0      | یکی از بی کیفیت ترین پدهایی بود که تابحال گرفتم قدرت جاذبه نداره چسب پشتش که اصلا نمیچسبه به سوتین پرز میده و پنبه های داخل پد گوله گوله میشه بعد از یروز استفاده از دیجی کالای عزیز تقاضا دارم توقف این محصول رو اعلام کنه و بجاش همون مارک وی و یا پنبه ریز رو موجود کنه . |
| 5      | بسیار گوشی مقرون به صرفه ای هست با این قیمت تصویر فول اچ دی و پشتیبانی از نسل چهار رو در هیچ برندی مشاهده نمیکنید رابط کاربری بسیار ساده ای داره و به راحتی باهاش آداپته میشید ولی ای کاش از مموری میکرو اس دی هم پشتیببانی میکرد این عدم پشتیبانی باعث میشه که این گوشی رو هر کسی نخره هر چند با یه فلش اوتیجی یا مبدلش مشکل رو میشه تا حدودی حل کرد . |
| 5      | خداییش من دارم راضیم نه خم میشه نه شاترش بد عمل میکنه دیجی ممنون . |

## Requirement
1. python 3
2. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install sklearn
```

## Usage
###production
```
execute run.py in production and open http://localhost in browser
```
###development
```bash
model_path = './models/svm.LinearSVC_classifier_for_predict_sample.pk'
vocabulary_path = "vector.pickel"
clf, vectorizer = makeReady(model_path, vocabulary_path)
predict(clf, vectorizer, '﻿از فرصت استفاده کنین بخرید . ')
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Hojat Tabkhi](https://)

production design: 
![production mode app view][logo]

[logo]: https://github.com/hojattabkhi/sentimentAnalysis/blob/master/production/templates/page.png "production mode app view"
