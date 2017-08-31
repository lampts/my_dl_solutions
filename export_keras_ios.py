import coremltools
from tensorflow.python.lib.io import file_io

# save model
    model_h5_name = 'model_acc_' + str(acc) + '.h5'
    model.save(model_h5_name)

    # save model.h5 on to google storage
    with file_io.FileIO(model_h5_name, mode='r') as input_f:
        with file_io.FileIO(logs_path + '/' + model_h5_name, mode='w+') as output_f:
            output_f.write(input_f.read())

            # reset session
            # Note: If this piece of code did help you to achieve your goal, please upvote my solution under:
            # https://stackoverflow.com/questions/41959318/deploying-keras-models-via-google-cloud-ml/44232441#44232441
            # Thank you so much :)
    k.clear_session()
    sess = tf.Session()
    k.set_session(sess)

    # disable loading of learning nodes
    k.set_learning_phase(0)

    # load model
    model = load_model(model_h5_name)
    config = model.get_config()
    weights = model.get_weights()
    new_Model = Sequential.from_config(config)
    new_Model.set_weights(weights)

    # export coreml model

    coreml_model = coremltools.converters.keras.convert(new_Model, input_names=['accelerations'],
                                                        output_names=['scores'])
    model_mlmodel_name = 'model_acc_' + str(acc) + '.mlmodel'
    coreml_model.save(model_mlmodel_name)

    # save model.mlmodel on to google storage
    with file_io.FileIO(model_mlmodel_name, mode='r') as input_f:
        with file_io.FileIO(logs_path + '/' + model_mlmodel_name, mode='w+') as output_f:
            output_f.write(input_f.read())

            # export saved model
            # Note: If this piece of code did help you to achieve your goal, please upvote my solution under:
            # https://stackoverflow.com/questions/41959318/deploying-keras-models-via-google-cloud-ml/44232441#44232441
            # Thank you so much :)
    export_path = logs_path + "/export"
    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(inputs={'accelerations': new_Model.input},
                                      outputs={'scores': new_Model.output})

    with k.get_session() as sess:
        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=[tag_constants.SERVING],
                                             signature_def_map={
                                                 signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
        builder.save()

