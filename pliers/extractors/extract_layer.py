''' Feature Vectors Extractor for a specific layer based on pre-traineed models'''


''' Extractor classes based on pre-trained models. '''

import os
import tempfile
import subprocess
import re
from pliers.extractors.models import TensorFlowInceptionV3Extractor
from pliers.extractors.base import ExtractorResult


class TensorFlowInceptionV3LayerExtractor(TensorFlowInceptionV3Extractor):

    ''' Labels objects in images using a pretrained Inception V3 architecture
     implemented in TensorFlow.

    Args:
        model_dir (str): path to save model file to. If None (default), creates
            and uses a temporary folder.
        data_url (str): URL to download model from. If None (default), uses
            the preset inception model (dated 2015-12-05) used in the
            TensoryFlow tutorials.
     '''

    _log_attributes = ('layer','model_dir', 'data_url')
    VERSION = '1.0'

    def __init__(self, layer='softmax:0', model_dir=None, data_url=None):

        super(TensorFlowInceptionV3LayerExtractor, self).__init__(model_dir,data_url)

        del self.num_predictions

        self.layer = layer


    def _extract(self,stim):
		# Adapted from def _extract(self, stim) in pliers'
		# models.py
        from pliers.external import tensorflow as tf
        tf_dir = os.path.dirname(tf.__file__)
        script = os.path.join(tf_dir,'extract_feature_image.py')

        with stim.get_filename() as filename:
            args = ' --image_file %s --model_dir %s --layer_name  %s' % \
            (filename, self.model_dir, self.layer)
            cmd=('python ' + script + args).split()
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            output, errors = process.communicate()
            hits = output.decode('utf-8').splitlines()

        values, features= [], []
        for i, h in enumerate(hits):
          m = re.search('\=\s([0-9\.]+)', h.strip())
          extraction = m.groups()
          values.append(float(extraction[0]))

        features.append("feature vector")

        feature_vector_extractor = ExtractorResult([0], stim, self, features=features)
        feature_vector_extractor._data = values

        return feature_vector_extractor