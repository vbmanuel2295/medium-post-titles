/**
 * This module is used for training a medium post classifier.
 *
 * 1. It loads data from the dataset/sample.csv
 * 2. It Translates the numeric class IDs to string based ones.
 * 3. It freezes the model's weights inside the ./medium-post-classifier.json
 */
const natural = require("natural");
const csv = require("csvtojson");

let persistModel = (classifier) => {
    return new Promise((resolve, reject) => {
        classifier.save("./medium-post-classifier.json", (err) => {
            if (err) {
                reject(err);
            } else {
                resolve();
            }
        });
    });
};

let loadData = () => {
    return new Promise((resolve) => {
        csv()
            .fromFile("./dataset/medium_post_titles.csv")
            .then((json) => {
                resolve(json);
            });
    });
};

(async () => {
    try {
        let dataset = await loadData();
        let classifier = new natural.BayesClassifier();

        dataset.forEach((i) => {

            classifier.addDocument(i.title, i.subtitle_truncated_flag);
        });

        classifier.train();

        await persistModel(classifier);
    } catch (err) {
        console.log(err.message);
        console.log(err.stack);
    }
})();