import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def proc():
    pass

def proc1():
    pass


labels = ['Bladder', 'External','Femur_L','Femur_R','PTV','Rectum.']
colors = ['b','g', 'r', 'c', 'm', 'y']

#input
test_name = 'cancer_pix2pix_semi_53_4_88_lambdaD.1'
gt_path = './datasets/cancer_semi_2/test/Dose'
pred_path = gt_path.replace('Dose', test_name)
contour_path = gt_path.replace('Dose', 'Images')
N_ORGANS = 6
nSlicePerPt = 64
H = 64
W = 64

#output:
#Dose_max = []
#Diff_Dose_max = [1.0, 1.2, 1.3]: N
#Dose_gt: [N, 64, 128, 128]
#Dose_pre

def save_dose_to_png( slice_gt, slice_pred, slice_contour, slice_path, slice_order):
    """
    :param slice_gt:
    :param slice_pred:
    :param slice_contour:
    :param slice_path:
    :param i: slice order
    :return:
    """

    vmin = 0
    vmax = .6
    contour_max_intensity = 15
    # save to png file how to make it color
    slice_contour_color = np.copy(slice_contour)
    for i in range(slice_contour.shape[0]):
        slice_contour_color[i, :, :] = slice_contour[i, :, :] * (i + 1)
    slice_contour_color = np.sum(slice_contour_color, axis=0, keepdims=False)  # make
    slice_contour_color = slice_contour_color / contour_max_intensity

    slice_by_slice = np.hstack((slice_contour_color, slice_gt, slice_pred))
    # plt.cla()
    plt.imshow(slice_by_slice, cmap='jet', vmin=vmin, vmax=vmax)
    slice_id = slice_order % nSlicePerPt
    pt_id = slice_order // nSlicePerPt
    plt.title('GT vs Pred: Pt:{}-{}'.format(pt_id, slice_id))

    slice_png_path = slice_path.replace('Dose', test_name)
    slice_png_path = slice_png_path.replace('npy', 'png')

    plt.savefig(slice_png_path)
    print slice_png_path
    plt.close()


    #### save each matrix separately
    plt.cla()
    # slice_id = slice_order % nSlicePerPt
    # pt_id = slice_order // nSlicePerPt
    # # plt.title('GT vs Pred: Pt:{}-{}'.format(pt_id, slice_id))
    #
    # slice_png_path = slice_path.replace('Dose', test_name)
    # slice_png_path = slice_png_path.replace('npy', 'png')

    #contour
    plt.imshow(slice_contour_color, cmap='jet', vmin=vmin, vmax=vmax)
    slice_contour_path = slice_png_path.replace('png','contour.png')
    plt.savefig(slice_contour_path)

    #gt
    plt.imshow(slice_gt, cmap='jet', vmin=vmin, vmax=vmax)
    slice_gt_path = slice_png_path.replace('png', 'gt.png')
    plt.savefig(slice_gt_path)

    plt.imshow(slice_pred, cmap='jet', vmin=vmin, vmax=vmax)
    slice_pred_path = slice_png_path.replace('png', 'pred.png')
    plt.savefig(slice_pred_path)

    plt.close()


    #### end of save

def get_array():
    slice_paths = sorted(glob.glob( os.path.join(gt_path, '*.npy')))
    N = len( slice_paths) /nSlicePerPt
    Dose_gt = np.zeros( (N*nSlicePerPt, H, W))
    Dose_pred = np.copy( Dose_gt)
    contour = np.zeros(( N * nSlicePerPt , N_ORGANS, H, W))


    for i, slice_path in enumerate(slice_paths):
        slice_gt = np.load( slice_path)
        slice_pred = np.load( slice_path.replace('Dose', test_name))
        slice_contour = np.load( slice_path.replace( 'Dose', 'Images'))

        save_dose_to_png( slice_gt, slice_pred, slice_contour, slice_path, i)

        #put to volume
        Dose_gt[i] = slice_gt
        Dose_pred[i] = slice_pred
        contour[i] = slice_contour

    Dose_gt = Dose_gt.reshape((-1, nSlicePerPt, H, W))
    Dose_pred = Dose_pred.reshape((-1, nSlicePerPt, H, W))
    contour = contour.reshape( (-1, nSlicePerPt, N_ORGANS, H, W))

    return Dose_gt, Dose_pred, contour


def get_array_clean(): #don't need to clean the noise due to prediction
    slice_paths = sorted(glob.glob( os.path.join(gt_path, '*.npy')))
    N = len( slice_paths) /nSlicePerPt
    Dose_gt = np.zeros( (N*nSlicePerPt, H, W))
    Dose_pred = np.copy( Dose_gt)
    contour = np.zeros(( N * nSlicePerPt , N_ORGANS, H, W))


    for i, slice_path in enumerate(slice_paths):
        slice_gt = np.load( slice_path)
        slice_pred = np.load( slice_path.replace('Dose', test_name))
        slice_contour = np.load( slice_path.replace( 'Dose', 'Images'))

        #clean when there are only 2 organs
        slice_contour_z = np.amax( slice_contour, axis= (1,2)) #decide if a layer has 1, very tricky
        if np.sum( slice_contour_z)<=2:
            slice_pred *= 0

        save_dose_to_png( slice_gt, slice_pred, slice_contour, slice_path, i)

        #put to volume
        Dose_gt[i] = slice_gt
        Dose_pred[i] = slice_pred
        contour[i] = slice_contour

    Dose_gt = Dose_gt.reshape((-1, nSlicePerPt, H, W))
    Dose_pred = Dose_pred.reshape((-1, nSlicePerPt, H, W))
    contour = contour.reshape( (-1, nSlicePerPt, N_ORGANS, H, W))

    return Dose_gt, Dose_pred, contour

def Dose_max_mean( gt, pred, contour, pt_i):
    gt_dose = gt[pt_i].squeeze()
    max_range = np.amax( gt_dose)

    pred_dose = pred[pt_i].squeeze()

    Dose_max = np.zeros( N_ORGANS)
    Dose_mean = np.zeros( N_ORGANS)

    for organ_i in range(N_ORGANS):
        vol_index = contour[pt_i,:,organ_i].squeeze().astype(bool)

        if np.sum( vol_index)==0:
            print organ_i, pt_i, 'Empty contour'
            continue

        gt_dose_vol = gt_dose[vol_index]
        pred_dose_vol = pred_dose[vol_index]

        dose_diff = pred_dose_vol - gt_dose_vol
        dose_diff_rel  =  abs(dose_diff) /(.5) #.5 is the prescription Dose
        Dose_max[organ_i] = np.amax( dose_diff_rel)
        Dose_mean[organ_i] = np.mean( dose_diff_rel)

    return Dose_max, Dose_mean

def Dose_max_mean_global( gt, pred, contour, pt_i):
    gt_dose = gt[pt_i].squeeze()
    max_range = np.amax( gt_dose)

    pred_dose = pred[pt_i].squeeze()

    Dose_max = np.zeros( N_ORGANS)
    Dose_mean = np.zeros( N_ORGANS)

    for organ_i in range(N_ORGANS):
        vol_index = contour[pt_i,:,organ_i].squeeze().astype(bool)

        if np.sum( vol_index)==0:
            print organ_i, pt_i, 'Empty contour'
            continue

        gt_dose_vol = gt_dose[vol_index]
        pred_dose_vol = pred_dose[vol_index]

        dose_diff_max = np.amax(pred_dose_vol) - np.amax(gt_dose_vol)
        dose_diff_mean = np.mean(pred_dose_vol) - np.mean(gt_dose_vol)
        dose_diff_max  =  abs(dose_diff_max) /(.5) #.5 is the prescription Dose
        dose_diff_mean = abs(dose_diff_mean) / (.5)  # .5 is the prescription Dose
        Dose_max[organ_i] = np.amax( dose_diff_max)
        Dose_mean[organ_i] = np.mean( dose_diff_mean)

    return Dose_max, Dose_mean

def dvh( gt, pred, contour, pt_i):
    '''
    contour: the contour or mask of the data
    :param gt:
    :param pred:
    :param contour:
    :param pt_i:
    :return:
    '''
    gt_dose = gt[pt_i].squeeze()
    max_range = np.amax( gt_dose) * 2

    pred_dose = pred[pt_i].squeeze()

    print "MSE: ", np.mean( np.square(gt_dose - pred_dose)) * 4
    print "MAE: ", np.mean( np.abs( gt_dose - pred_dose)) * 2

    for organ_i in range(N_ORGANS):
        vol_index = contour[pt_i,:,organ_i].squeeze().astype(bool)

        gt_dose_vol = gt_dose[vol_index] *2
        pred_dose_vol = pred_dose[vol_index] *2
        # dose_hist = np.histogram( dose_vol, bins=100)
        plt.hist( gt_dose_vol, bins= 100, range=(0, max_range), normed=1, cumulative=-1,
                  histtype= 'step', color=colors[organ_i], linestyle = 'solid', label=labels[organ_i])

        plt.hist( pred_dose_vol, bins=100, range=(0, max_range), normed=1, cumulative=-1,
                  histtype='step', color=colors[organ_i], linestyle = 'dashed')

    # legend = plt.legend(loc='best')
    plt.title("Plan: Solid, Predict: Dashed \n labels = ['Bladder', 'External','Femur_L','Femur_R','PTV','Rectum.'] \n colors = ['b','g', 'r', 'c', 'm', 'y']")
    #plt.show()
    plt.savefig( pred_path + '/DVH_' + str(pt_i) + '.png')
    plt.close()




def main():
    Dose_gt, Dose_pred, contour = get_array()
    N = len( Dose_gt)
    Dose_max = np.zeros((N, N_ORGANS))
    Dose_mean = np.zeros((N, N_ORGANS))

    for pt_i in range( len( Dose_pred)):
        dvh( Dose_gt, Dose_pred, contour, pt_i)
        Dose_max[pt_i], Dose_mean[pt_i] = Dose_max_mean_global( Dose_gt, Dose_pred, contour, pt_i)

    dose_ave1 = Dose_max.mean(axis = 0)
    dose_ave2 = Dose_mean.mean(axis=0)
    Dose_max = np.vstack((Dose_max, dose_ave1))
    Dose_mean = np.vstack((Dose_mean, dose_ave2))
    np.savetxt(pred_path+'/Max_dose.csv', Dose_max, delimiter=',',fmt= '%.3f', header=','.join(labels))
    np.savetxt(pred_path+'/Mean_dose.csv', Dose_mean, delimiter=',',fmt= '%.3f', header=','.join(labels))
if __name__ == '__main__':
    main()