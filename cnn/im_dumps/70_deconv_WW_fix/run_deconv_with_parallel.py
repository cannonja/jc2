
import bprofile
import click
import job_stream
import job_stream.inline as inline
from mr.datasets.tower import StanfordTower, TowerScaffold
from mr.learn.converger import Converger
from mr.learn.convolve import ConvolveLayer, DeconvolveLayer, PassthroughLayer
from mr.learn.scaffold import Scaffold
from mr.learn.supervised.perceptron import Perceptron
from mr.learn.unsupervised.lca import Lca
from mr.learn.unsupervised.lcaSpikingWoodsAnalytical import LcaSpikingWoodsAnalyticalInhibition
import numpy as np
import os
import skvideo.io

from video_util import VideoHelper

folder = '/stash/tlab/datasets/Tower'
file_pre = 'Neovision2-Training-Tower-'
num_classes = 5

train_folders = [1,2,3,4,5]  # 1,2,3,4,5
test_folders = [13, 14]  # 13,14

if False:
    folder = '/u/wwoods/Downloads/NeoVisionTemp'
    train_folders = [1]
    test_folders = [1]

@click.command()
@click.option('--output', default='example')
@click.option('--epochs', default=500)
@click.option('--mini-epochs', default=4)
@click.option('--width', default=70)
@click.option('--tau', default=1)
def main(**kwargs):
    def h(msg):
        print("."*79)
        print("."*79)
        print(msg)
        print("."*79)
        print("."*79)
    def get_folders(folders, type):
        return [os.path.join(folder, type, str(i), '{}{}'.format(file_pre,
            str(i).zfill(3))) for i in folders]
    def get_csv(folders, type):
        return [f+'.csv' for f in get_folders(folders, type)]
    train_vid = get_folders(train_folders, 'train')
    test_vid = get_folders(test_folders, 'test')
    train_csv = get_csv(train_folders, 'train')
    test_csv = get_csv(test_folders, 'test')

    # A person is, by Jack's estimate, 30 pixels by something.  To pick up
    # people, we therefore need to include at least 30 pixels.  So, for
    # W=35, that'a W*30/1920.  Include 5 times this size.
    convSize = int(round(5 * kwargs['width'] * 30 / 1920))
    convStride = convSize
    #convSize = convStride = 10
    if job_stream.getRank() == 0:
        h("Convolving size={}, stride={}".format(convSize, convStride))

    deconvSize = convSize * 3
    deconvStride = convSize
    if job_stream.getRank() == 0:
        h("Deconvolving size={}, stride={}".format(deconvSize, deconvStride))

    # Convert the image dimensions sanely.  Convolution layer requires
    # that (imW - convSize) % convStride == 0.
    nw = kwargs['width']
    nh = round(nw * 1088 / 1920)

    nw = int(round((nw - convSize) / convStride)) * convStride + convSize
    nh = int(round((nh - convSize) / convStride)) * convStride + convSize

    if job_stream.getRank() == 0:
        h("Loading videos as {}".format((nw, nh)))
    #with bprofile.BProfile("profile.png"):
    tower = StanfordTower(nw, nh, train_vid, test_vid, train_csv,
            test_csv, kwargs['tau'])
    train, test, vp = tower.split()

    model2 = TowerScaffold()

    # Use job_stream for parallelism... Basically, we have a few loops (frames):
    # 1. Over epochs
    #   2a. Over training
    #   2b. Over testing

    def getSplits(arr, cpus):
        """Given a number of cpus, return how many splits there should actually
        be for a given array.
        """
        # Empirically, communication bandwidth allows for 8 at 140 width
        minExp = max(1, int(8 * (140. / nw) ** 2))
        # Note that we allow cpus*2 so that faster processing units can do more
        # of the work, if the communication overhead is proportionally low
        # enough
        return np.array_split(arr, max(1, min(cpus*2, len(arr) // minExp)))


    with inline.Work() as w:
        @w.init
        def getFirstScaf():
            scaf = Scaffold()
            cl = ConvolveLayer(
                    #PassthroughLayer(),
                    Lca(convSize*convSize*3 // 2, simEpsilon=0.1),
                    #LcaSpikingWoodsAnalyticalInhibition(convSize*convSize*3//4, rfAvg=0.45),
                    vp, convSize=convSize,
                    convStride=convStride)
            cl.init(len(train[0][0]), None)
            scaf.layer(cl)
            dl = DeconvolveLayer(Perceptron(),
                    vp[:2] + (cl.visualParams[2], num_classes),
                    convSize=cl.convSize, convStride=cl.convStride,
                    deconvSize=convSize, deconvStride=convStride)
            scaf.layer(dl)
            scaf.init(len(train[0][0]), len(train[1][0]))
            return scaf


        @w.frame
        def byEpochStart(store, first):
            if not hasattr(store, 'init'):
                cpus = inline.job_stream.getCpuCount()
                print("Using training parallelism: {}".format(len(getSplits(
                        range(len(train[0]) // kwargs['mini_epochs']), cpus))))
                print("Using testing parallelism: {}".format(len(getSplits(
                        range(len(test[0])), cpus))))
                store.init = True
                store.epoch = -1
                store.net = first

            store.epoch += 1
            if store.epoch >= kwargs['epochs']:
                return
            h("Epoch {}".format(store.epoch))
            return store.net

        def trainLoop():
            """Adds the training loop to the job_stream.  Since this form of
            parallelism is basically batching, we want to be sure that we have
            a number of "mini epochs" per epoch.
            """
            MINI_EPOCHS = kwargs['mini_epochs']
            # NOTE - multiprocessing CANNOT be enabled for this frame because
            # the Converger() object won't allow it.
            @w.frame(emit=lambda store: store.net, useMultiprocessing=False)
            def doTrainStart(store, net):
                if not hasattr(store, 'init'):
                    store.init = True
                    store.net = net
                    store.c = Converger()
                    store.c.track()
                    store.c.init(net)
                    store.me = 0

                if store.me != 0:
                    # Merge last mini epoch's changes
                    store.c.commit(store.net)
                if store.me >= MINI_EPOCHS:
                    # All done
                    return

                cpus = inline.job_stream.getCpuCount()
                s = len(train[0]) * store.me // MINI_EPOCHS
                e = len(train[0]) * (store.me+1) // MINI_EPOCHS
                train_idx = getSplits(range(s, e), cpus)

                store.me += 1

                print("Training... {}-{}/{}".format(s, e, len(train[0])))
                return inline.Multiple([
                        inline.Args(store.net, t)
                        for t in train_idx
                        if t.shape[0] != 0])
            @w.job
            def doTrain(net, train_idx):
                net.partial_fit(train[0][train_idx], train[1][train_idx])
                return net
            @w.frameEnd
            def doTrainEnd(store, next):
                store.c.update(next)
        trainLoop()

        def testLoop():
            """Assembles all frame predictions, and produces a score and
            example video.
            """
            @w.frame(emit=lambda store: store.net)
            def doTestStart(store, net):
                if not hasattr(store, 'init'):
                    print("Testing...")
                    store.init = True
                    store.net = net
                    store.preds = []
                    cpus = inline.job_stream.getCpuCount()
                    test_idx = getSplits(range(len(test[0])), cpus)
                    return inline.Multiple([
                            inline.Args(store.net, t)
                            for t in test_idx
                            if t.shape[0] != 0])

                # End of loop, merge predictions in correct order
                store.preds.sort()
                xP = np.concatenate([preds for _, preds in store.preds])

                # Print scores
                iu_scores = model2.calc_iu(xP, test[1], num_classes)
                mm_scores = model2.calc_mm(xP, test[1], num_classes)
                print("IU score: {}".format({tower.rclasses.get(k, k): v
                        for k, v in iu_scores.items()}))
                print("MM score: {}".format({tower.rclasses.get(k, k): v
                        for k, v in mm_scores.items()}))

                # Make a video
                # car, truck, bus, person, cyclist
                CLS_COLOR = tower.colors
                with VideoHelper(
                        "{}.mp4".format(kwargs['output']),
                        fps=29.97,
                        frameSize=(vp[0]*2, vp[1])) as vh:
                    for i in range(len(test[0])):
                        with vh.frame() as buffer:
                            buffer.blit_flat_float[:vp[0], :] = test[0][i]
                            for j in range(num_classes):
                                buffer.blit_flat_float_mono_as_alpha[
                                        :vp[0], :, CLS_COLOR[j], 0.5] = (
                                                test[1][i][j::num_classes])
                                buffer.blit_flat_float_mono_as_alpha[
                                        vp[0]:, :, CLS_COLOR[j]] = (
                                                xP[i, j::num_classes])


            @w.job
            def doTest(net, idx):
                return (idx[0], net.predict(test[0][idx]))

            @w.frameEnd
            def doTestEnd(store, next):
                store.preds.append(next)
        testLoop()

        @w.frameEnd
        def byEpochEnd(store, next):
            store.net = next


if __name__ == '__main__':
    main()

