

def runAll():
    import settings
    q_vals = ["q1", "q2", "q3","q4" ]
    models = ["resnet18"]
    datasets = ["places365", "imagenet"]
    version = "v4"
    for dataset in datasets:
        settings.DATASET = dataset
        for model in models:
            for q in q_vals:
                settings.MODEL = model
                settings.Q = q
                settings.OUTPUT_FOLDER = "result/"+ version + "_"+model+"_"+dataset+"_"+q
                settings.INDEX_FILE = "index_" + q + '.csv'
                settings.DATA_DIRECTORY = "dataset/broden1_224_"+ q
                print("Getting data from: " + settings.DATA_DIRECTORY)
                print("Using index file: " + settings.INDEX_FILE)
                print("Model file: " + settings.MODEL_FILE)
                main()
                print("Finished {} on model {}".format(q, model))

def main(): 
    import settings
    from loader.model_loader import loadmodel
    from feature_operation import hook_feature,FeatureOperator
    from visualize.report import generate_html_summary
    from util.clean import clean

    fo = FeatureOperator()
    model = loadmodel(hook_feature)
    print(settings.Q)
    ############ STEP 1: feature extraction ###############
    features, maxfeature = fo.feature_extraction(model=model)

    for layer_id,layer in enumerate(settings.FEATURE_NAMES):
        ############ STEP 2: calculating threshold ############
        thresholds = fo.quantile_threshold(features[layer_id],savepath="quantile.npy")

        ############ STEP 3: calculating IoU scores ###########
        tally_result = fo.tally(features[layer_id],thresholds,savepath="tally.csv")

            ############ STEP 4: generating results ###############
        generate_html_summary(fo.data, layer,
                            tally_result=tally_result,
                            maxfeature=maxfeature[layer_id],
                            features=features[layer_id],                                                                                thresholds=thresholds)
        if settings.CLEAN:
            clean()


runAll()
