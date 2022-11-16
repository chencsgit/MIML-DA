import torch.nn as nn
from maml.models.functions import ReverseLayerF
import torch

class DSN(nn.Module):
    def __init__(self, code_size=100, n_class=10, inputsize = 400, outputsize = 200):
        super(DSN, self).__init__()
        self.code_size = code_size

        ##########################################
        # private source encoder
        ##########################################

        self.source_encoder_conv = nn.Sequential()
        output = int(outputsize*2)
        inputShared_decoder_conv = output
        self.source_encoder_conv.add_module('layer{}_linear', nn.Linear(in_features=inputsize, out_features=output))

        self.source_encoder_fc = nn.Sequential()
        self.source_encoder_fc.add_module('fc_pse3', nn.Linear(in_features=7 * 7 * 64, out_features=code_size))
        self.source_encoder_fc.add_module('ac_pse3', nn.ReLU(True))

        #########################################
        # private target encoder
        #########################################

        self.target_encoder_conv = nn.Sequential()
        self.target_encoder_conv.add_module('layer{}_linear', nn.Linear(in_features=inputsize, out_features=output))

        self.target_encoder_fc = nn.Sequential()
        self.target_encoder_fc.add_module('fc_pte3', nn.Linear(in_features=7 * 7 * 64, out_features=code_size))
        self.target_encoder_fc.add_module('ac_pte3', nn.ReLU(True))

        ################################
        # embedding
        ################################
        self.embedding512 = nn.Sequential()
        output2 = int(output*2)
        self.embedding512.add_module('512', nn.Linear(in_features=output, out_features= output2))


        ################################
        # shared encoder (dann_mnist)
        ################################
        output = int(outputsize / 2)
        inputShared_decoder_conv = output + inputShared_decoder_conv
        self.shared_encoder_conv = nn.Sequential()
        self.shared_encoder_conv.add_module('layer{}_linear', nn.Linear(in_features=inputsize, out_features=output))

        self.shared_encoder_fc = nn.Sequential()
        self.shared_encoder_fc.add_module('fc_se3', nn.Linear(in_features=7 * 7 * 48, out_features=code_size))
        self.shared_encoder_fc.add_module('ac_se3', nn.ReLU(True))
        ################################
        # embedding
        ################################
        self.embedding128 = nn.Sequential()
        output2 = int(outputsize)
        self.embedding128.add_module('128', nn.Linear(in_features=output, out_features=output2))

        # classify 10 numbers
        self.shared_encoder_pred_class = nn.Sequential()
        self.shared_encoder_pred_class.add_module('fc_se4', nn.Linear(in_features=code_size, out_features=100))
        self.shared_encoder_pred_class.add_module('relu_se4', nn.ReLU(True))
        self.shared_encoder_pred_class.add_module('fc_se5', nn.Linear(in_features=100, out_features=n_class))

        self.shared_encoder_pred_domain = nn.Sequential()
        halfOut = int(outputsize/2)
        self.shared_encoder_pred_domain.add_module('fc_se6', nn.Linear(in_features=outputsize, out_features=halfOut))
        self.shared_encoder_pred_domain.add_module('relu_se6', nn.ReLU(True))

        # classify two domain
        self.shared_encoder_pred_domain.add_module('fc_se7', nn.Linear(in_features=halfOut, out_features=2))

        ######################################
        # shared decoder (small decoder)
        ######################################

        self.shared_decoder_fc = nn.Sequential()
        self.shared_decoder_fc.add_module('fc_sd1', nn.Linear(in_features=code_size, out_features=588))
        self.shared_decoder_fc.add_module('relu_sd1', nn.ReLU(True))

        self.shared_decoder_conv = nn.Sequential()
        self.shared_decoder_conv.add_module('layer{}_linear', nn.Linear(in_features=inputShared_decoder_conv, out_features=inputsize))




    def forward(self, input_data, mode, rec_scheme, p=0.0):

        result = []

        if mode == 'source':

            # source private encoder
            #private_feat = self.source_encoder_conv(input_data)
            #private_feat = private_feat.view(-1, 64 * 7 * 7)
            #private_code = self.source_encoder_fc(private_feat)
            private_code = self.source_encoder_conv(input_data)
        elif mode == 'target':

            # target private encoder
            #private_feat = self.target_encoder_conv(input_data)
            #private_feat = private_feat.view(-1, 64 * 7 * 7)
            #private_code = self.target_encoder_fc(private_feat)
            private_code =self.target_encoder_conv(input_data)
        result.append(private_code)

        # shared encoder
        #shared_feat = self.shared_encoder_conv(input_data)
        #shared_feat = shared_feat.view(-1, 48 * 7 * 7)
        #shared_code = self.shared_encoder_fc(shared_feat)
        shared_code = self.shared_encoder_conv(input_data)
        result.append(shared_code)

        #reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = 0#self.shared_encoder_pred_domain(reversed_shared_code)
        result.append(domain_label)

        if mode == 'source':
            #class_label = self.shared_encoder_pred_class(shared_code)
            class_label = []
            result.append(class_label)

        # shared decoder

        if rec_scheme == 'share':
            union_code = shared_code
        elif rec_scheme == 'all':
            #union_code = private_code + shared_code
            union_code = torch.cat((private_code, shared_code), 1)
            #union_code =0
        elif rec_scheme == 'private':
            union_code = private_code

        #rec_vec = self.shared_decoder_fc(union_code)
        #rec_vec = rec_vec.view(-1, 3, 14, 14)

        rec_code = self.shared_decoder_conv(union_code)
        result.append(rec_code)

        embedding128 = self.embedding128(shared_code)
        embedding512 = self.embedding512(private_code)
        result.append(embedding128)
        result.append(embedding512)
        return result





