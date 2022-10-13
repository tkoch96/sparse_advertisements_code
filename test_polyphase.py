import numpy as np, matplotlib.pyplot as plt, time
import scipy.signal as sg
from scipy.ndimage import gaussian_filter1d as gfilt


def polyphase_filt(data, filt, decimate_by):
    ### Implements fftconv + downsample with polyphase

    ## since we downsample and since distributions are often bernoulli-ish, 
    ## we need to smooth everything with gaussian filters
    data = gfilt(data, 1, axis=0)
    filt = gfilt(filt, 1, axis=0)
    if data.shape[0] % decimate_by != 0:
        data = data[:(data.shape[0] - data.shape[0]%decimate_by),:]
    if filt.shape[0] % decimate_by != 0:
        filt = filt[:(filt.shape[0] - filt.shape[0]%decimate_by),:]
    # stack data, reversing row ordering past the first
    head_data_stream = np.expand_dims(data[::decimate_by,:],axis=0)
    tail_data_streams = np.asarray([np.asarray(data[0+i::decimate_by,:])  for i in range(1,decimate_by)])
    # data_streams = np.vstack((head_data_stream, tail_data_streams[:,::-1,:]))
    data_streams = np.concatenate([head_data_stream, tail_data_streams[:,::-1,:]], axis=0)
    # stack filters
    filt = np.expand_dims(filt,axis=0)
    filter_streams = [np.expand_dims(filt[0,0+i::decimate_by,:],axis=0) for i in range(decimate_by)]
    filter_streams = np.concatenate(filter_streams, axis=0)
    filtered_data_streams = sg.fftconvolve(filter_streams, data_streams, axes=1)
    s = filtered_data_streams.shape
    new_filtered_data_streams = np.zeros((s[0], s[1]+1,s[2]))
    for i in range(decimate_by):
        if i == 0:
            new_filtered_data_streams[i,:-1,:] = filtered_data_streams[i,:,:] # postpend zero
        else:
            new_filtered_data_streams[i,1:,:] = filtered_data_streams[i,:,:] # prepend zero
    filtered_data = np.sum(new_filtered_data_streams, axis=0)
    return filtered_data

def not_polyphase_filt(data, filt, decimate_by):
    #### Implements fftconv and mean over bins by decimate_by

    if data.shape[0] % decimate_by != 0:
        data = data[:(data.shape[0] - data.shape[0]%decimate_by),:]
    if filt.shape[0] % decimate_by != 0:
        filt = filt[:(filt.shape[0] - filt.shape[0]%decimate_by),:]

    filtered_data_streams = sg.fftconvolve(data, filt, axes=0)
    if filtered_data_streams.shape[0] % decimate_by != 0:
        leftover = decimate_by - (filtered_data_streams.shape[0] % decimate_by)
        filtered_data_streams = np.concatenate([filtered_data_streams, 
            np.zeros((leftover, filtered_data_streams.shape[1]))], axis=0)
    filtered_data = filtered_data_streams.reshape((filtered_data_streams.shape[0]//decimate_by,
        decimate_by,filtered_data_streams.shape[1],1)).mean(-1).mean(1)

    return filtered_data

def sum_pdf_new(px):
    ## Calculates pdf of sum of RVs, each column is a pdf of a RV
    ## so px is n_bins x n_RVs
    ## output n_bins is n_bins


    l = px.shape[0]
    nout = px.shape[1]//2
    if px.shape[1] % 2 == 0:
        output = np.zeros((l, nout))
    else:
        output = np.zeros((l, nout+1))
        output[:,-1] = px[:,-1]
    ## approximates actual convolution, since we efficiently implement operations using polyphase filters
    # output[0:2*(l//2),0:nout] = polyphase_filt(px[:,:nout*2:2], px[:,1:nout*2:2], 2)
    ## performs actual convolution, a bit slower
    output[0:2*(l//2),0:nout] = not_polyphase_filt(px[:,:nout*2:2], px[:,1:nout*2:2], 2)
    output = output.clip(0,np.inf)
    if output.shape[1] >= 2:
        output = sum_pdf_new(output)
    return output / np.sum(output,axis=0)

def sum_pdf_old(px):
    l_post_pad = (px.shape[1] + 1) * px.shape[0]
    n_fft = px.shape[0] + l_post_pad
    n_fft = int(2**(np.ceil(np.log2(n_fft)))) # make it a nice power of 2
    # print("nfft: {}".format(n_fft))
    Px = np.fft.fft(px,n=n_fft,axis=0) # pop into freq
    Psumx = np.prod(Px,axis=1) # convolution in time domain is product in frequency
    psumx = np.real(np.fft.ifft(Psumx))

    # downsample by summing
    psumx_out = np.zeros((px.shape[0]))
    for i in range(px.shape[0]):
        psumx_out[i] = np.sum(psumx[i*px.shape[1]:(i+1)*px.shape[1]])
    return psumx_out

if __name__ == "__main__":
    n_users = 1000
    n_bins = 100
    px = np.random.uniform(size=(n_bins,n_users))
    px = px / np.sum(px,axis=0)


    # Time comparison
    ts = time.time()
    out_old = sum_pdf_old(px)
    print("Old took {}".format(time.time() - ts))
    ts=time.time()
    out_new = sum_pdf_new(px)
    print("New took {}".format(time.time() - ts))
    # output validity comparison (both code working and downsample nonsense)
    f,ax = plt.subplots(2)
    ax[0].plot(out_old)
    ax[1].plot(out_new)
    plt.show()
