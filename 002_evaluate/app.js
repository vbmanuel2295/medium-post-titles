const natural = require("natural");
const csv = require("csvtojson");
const naiveBayes = natural.BayesClassifier;
const MODEL_PATH = "../001_train/medium-post-classifier.json";

let loadFrozenModel = () => {
    return new Promise((resolve, reject) => {
        naiveBayes.load(MODEL_PATH, null, (err, classifier) => {
            if (err) {
                reject(err);
            }

            resolve(classifier);
        });
    });
};

let loadData = () => {
    return new Promise((resolve) => {
        csv()
            .fromFile("../001_train/dataset/medium_post_titles.csv")
            .then((json) => {
                resolve(json);
            });
    });
};


(async () => {
    try {
        let [classifier, data] = await Promise.all([loadFrozenModel(), loadData()]);
        let results = data.map((i, idx) => {

            let predicted = classifier.classify(i.title);

            console.clear();
            console.log(`Inferring Item ${idx + 1} out of ${data.length}`);

            return {
                text: i.title,
                actual: i.subtitle_truncated_flag,
                predicted,
                isCorrect: i.subtitle_truncated_flag === predicted,
            };
        });
        let accurateHit = results.filter((i) => i.isCorrect);
        let missedItems = results.filter((i) => !i.isCorrect);
        let accuracy = (accurateHit.length / data.length) * 100;
        let errorRate = (missedItems.length / data.length) * 100;

        console.log(`Accurate: ${accurateHit.length} items`);
        console.log(`Missed: ${missedItems.length} items`);
        console.log(`Accuracy: ${accuracy.toFixed(2)}`);
        console.log(`Error Rate: ${errorRate.toFixed(2)}`);
    } catch (err) {
        console.log(err.message);
        console.log(err.stack);
    }
})();