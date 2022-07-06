# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:12:37 2019

@author: cm
"""


import os
import sys
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import tensorflow as tf
from networks import NetworkAlbertTextCNN
from classifier_utils import get_feature_test,id2label
from hyperparameters import Hyperparamters as hp
 
          

class ModelAlbertTextCNN(object,):
    """
    Load NetworkAlbert TextCNN model
    """
    def __init__(self):
        self.albert, self.sess = self.load_model()
    @staticmethod
    def load_model():
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                albert =  NetworkAlbertTextCNN(is_training=False)
                saver = tf.train.Saver()  
                sess.run(tf.global_variables_initializer())
                checkpoint_dir = os.path.abspath(os.path.join(pwd,hp.file_load_model))
                print (checkpoint_dir)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                saver.restore(sess, ckpt.model_checkpoint_path)
        return albert,sess

MODEL = ModelAlbertTextCNN()
print('Load model finished!')


def get_label(sentence):
    """
    Prediction of the sentence's label.
    """
    feature = get_feature_test(sentence)
    fd = {MODEL.albert.input_ids: [feature[0]],
          MODEL.albert.input_masks: [feature[1]],
          MODEL.albert.segment_ids:[feature[2]],
          }
    prediction = MODEL.sess.run(MODEL.albert.predictions, feed_dict=fd)[0]   
    print(prediction)
    # r=[]
    # for i in range(len(prediction)):
    #     if prediction[i]!=0.0:
    #         r.append(id2label(i))
    # return r
    return [id2label(l) for l in np.where(prediction==1)[0] if l!=0]      


if __name__ == '__main__':
    # Test
    sentences = ['落腳台灣超過30年的帝亞吉歐，長期實踐企業社會責任、推動「KEEP WALKING 夢想資助計畫」也來到第18屆，「相信很少有外商公司能在一個地方如此長時間落實CSR理念」台灣帝亞吉歐表示。作為全球知名酒精性飲料領導廠商，帝亞吉歐不只要教育消費者「理性飲酒」的重要性，同時也必須肩負起推動「社會共融」及ESG「環境永續」等三大目標的重責大任，而夢想資助計畫正是從社會共融中延伸出來的內容，其目的正是為了在台灣這塊土地上打造一個別具意義的資源平台，鼓勵在這塊土地上的人們都能勇敢的築夢踏實、創造出更豐富的生命力。特別在這兩年全球受到疫情的衝擊，帝亞吉歐也重新思考這個夢想資助計畫還能如何發揮自身的影響力，因此首度規劃了『餐飲服務振興計畫』，而這背後其實隱含著許多帝亞吉歐深思熟慮的想法。首屆餐飲服務振興計畫，兩位得主展現不同的影響力餐飲業是疫情下的海嘯第一排，特別是帝亞吉歐的商品與許多餐飲業都有密切合作，他們比任何人都還要了解其中的衝擊，因此積極與政府單位交換意見，特別以「夢想，逆境重生」作為主題，目的就是希望在迎接疫後復甦之際，能陪伴這些餐飲界的夢想家一同從逆境中破繭而出，勇敢迎向未來的挑戰，用創新思維繼續實踐夢想。今年KEEP WALKING夢想資助計畫強調賦權與賦能及夢想的永續性與包容性，依「對社會之正面影響性」、「夢想發展性」 、「夢想前瞻性」及「夢想籌備積極性」等四項評分標準進行評選，最終選出12位年度夢想得主，其中包括2名餐飲服務振興計畫得獎者。有別於以往主要針對夢想資助計畫的得主進行獎金補助，帝亞吉歐表示今年特地積極規劃了技能培訓課程、商管或實務操作等賦能計畫，希望能教育並培力更多夢想資助計畫的參賽者，「我們不該是給他們魚吃、而是要教他們如何釣魚，」帝亞吉歐從作法上落實永續想法，藉由提升參賽者的實力，即便未能得獎、他們依舊能在市場上繼續實踐對餐飲的熱情跟夢想。最終餐飲服務振興計畫破格入選兩位各有特色、夢想主題截然不同的得獎者，分別是調酒師唐賢懿跟他的夥伴李玹毅的「Alloy Inception計畫」，及對味廚房料理實驗室研發主廚洪昭勝的「永續飲食大學學生餐廳夢想改造計畫」。']
    for sentence in sentences:
         print(get_label(sentence)) #sentence,
    



    
    
    
    
