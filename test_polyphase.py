import numpy as np, matplotlib.pyplot as plt, time
import scipy.signal as sg
from constants import *
from scipy.ndimage import gaussian_filter1d as gfilt


def rescale_pdf(x, px, renorm):
    ### Rescales (x,px) to be from x[0] / renorm -> x[-1] / renorm
    ### but we truncate them to retain the same spacing as in x
    ### i.e., we focus in on the region with probability, which we assume
    ### to be much smaller than x[0] -> x[-1]
    n_out = len(px)
    delta = x[1] - x[0]
    n_needed = np.maximum(n_out,int(np.ceil(n_out / renorm)))
    rescaled_full_x = x[0] / renorm + delta * np.arange(n_needed)

    full_rescaled_px = np.zeros(n_needed)
    prob_concentration = np.where(px>0.0001)[0]
    for ind in prob_concentration:
        full_rescaled_px[int(ind/renorm)] += px[ind]

    prob_concentration = np.where(full_rescaled_px>0)[0]
    minx,maxx = prob_concentration[0], prob_concentration[-1]+1
    if maxx-minx < n_out:
        minx = np.maximum(maxx-n_out,0)
        if maxx-minx < n_out:
            maxx = minx + n_out
    elif maxx-minx > n_out:
        minx = maxx - n_out
    rescaled_px = full_rescaled_px[minx:maxx]
    if minx > 0:
        rescaled_px[0] += np.sum(full_rescaled_px[0:minx])
    rescaled_x = rescaled_full_x[minx:maxx]

    return rescaled_x, rescaled_px

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

def decimate_fftconv(data, filt, decimate_by):
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

def clip_fftconv(data, filt, clip_by):
    #### Implements fftconv and mean over bins by decimate_by
    #### Clips upper "clip_by" portion of the array, accumulating the leftover probability
    ### to that last bin

    if data.shape[0] % clip_by != 0:
        data = data[:(data.shape[0] - data.shape[0]%clip_by),:]
    if filt.shape[0] % clip_by != 0:
        filt = filt[:(filt.shape[0] - filt.shape[0]%clip_by),:]

    filtered_data_streams = sg.fftconvolve(data, filt, axes=0)
    if filtered_data_streams.shape[0] % clip_by != 0:
        leftover = clip_by - (filtered_data_streams.shape[0] % clip_by)
        filtered_data_streams = np.concatenate([filtered_data_streams, 
            np.zeros((leftover, filtered_data_streams.shape[1]))], axis=0)
    nr,nc = filtered_data_streams.shape
    removing = filtered_data_streams[0:nr//clip_by,:]
    filtered_data_streams[nr//clip_by,:] += np.sum(removing,axis=0)
    filtered_data = filtered_data_streams[nr//clip_by:,:]

    return filtered_data

def fixed_point_fftconv(x1, data, x2, filt,**kwargs):
    #### Implements fftconv and mean over bins
    #### Finds output array of same size as inputs that contains most of the probability
    #### returns filtered (summed) pmfs and associated x regions

    clip_by = 2
    if data.shape[0] % clip_by != 0:
        data = data[:(data.shape[0] - data.shape[0]%clip_by),:]
    if filt.shape[0] % clip_by != 0:
        filt = filt[:(filt.shape[0] - filt.shape[0]%clip_by),:]

    filtered_data_streams = sg.fftconvolve(data, filt, axes=0)
    if filtered_data_streams.shape[0] % clip_by != 0:
        leftover = clip_by - (filtered_data_streams.shape[0] % clip_by)
        filtered_data_streams = np.concatenate([filtered_data_streams, 
            np.zeros((leftover, filtered_data_streams.shape[1]))], axis=0)
    nr,nc = filtered_data_streams.shape

    prob_concentration = np.where(filtered_data_streams>.0001)

    DEFAULT_VAL = 100000000
    by_column_concentration = {i:{'vals':[],'min': DEFAULT_VAL, 'max': -DEFAULT_VAL} for i in range(filtered_data_streams.shape[1])}
    # for each user (y) find first and last index that user has a significant amount of probability
    for x,y in zip(*prob_concentration):
        # by_column_concentration[y]['min'] = np.minimum(x,by_column_concentration[y]['min'])
        # by_column_concentration[y]['max'] = np.maximum(x+1,by_column_concentration[y]['max'])
        by_column_concentration[y]['vals'].append(x)
    for y in by_column_concentration:
        by_column_concentration[y]['min'] = np.min(by_column_concentration[y]['vals'])
        by_column_concentration[y]['max'] = np.max(by_column_concentration[y]['vals'])+1

    allowable_amt = data.shape[0]
    new_filtered_data_streams = np.zeros(data.shape) # downsampled
    x_mins = x1[0,:] + x2[0,:] # x return array
    x_maxs = x1[-1,:] + x2[-1,:]
    full_x = np.zeros(filtered_data_streams.shape)
    for i in range(filtered_data_streams.shape[1]):
        full_x[:-1,i] = np.linspace(x_mins[i], x_maxs[i], num=filtered_data_streams.shape[0]-1)
    ret_x = np.zeros(data.shape)

    for i in range(filtered_data_streams.shape[1]):
        # Find ranges to keep
        minx,maxx = by_column_concentration[i]['min'], by_column_concentration[i]['max']
        total_range = maxx-minx
        if total_range < allowable_amt:
            leftover = allowable_amt - total_range
            minx -= leftover
            minx = np.maximum(0,minx)
            total_range = maxx-minx
            if total_range < allowable_amt:
                leftover = allowable_amt - total_range
                maxx += leftover
        elif total_range > allowable_amt:
            ## need to chop something off, chop off the negatives since those are less likely
            minx = maxx - allowable_amt

        # Chop off unused data
        removing_lower = filtered_data_streams[0:minx,i]
        removing_upper = filtered_data_streams[maxx:,i]
        if removing_lower.shape[0] > 0:
            filtered_data_streams[minx,i] += np.sum(removing_lower)
        if removing_upper.shape[0] > 0:
            filtered_data_streams[maxx,i] += np.sum(removing_upper)
        new_filtered_data_streams[:,i] = filtered_data_streams[minx:maxx,i]
        ret_x[:,i] = full_x[minx:maxx,i] 

    return ret_x, new_filtered_data_streams

def sum_pdf_new(px):
    ## Calculates pdf of sum of RVs, each column is a pdf of a RV
    ## so px is n_bins x n_RVs
    ## output n_bins, since we downsample by clipping

    if px.shape[1] == 1:
        return px
    l = px.shape[0]
    nout = px.shape[1]//2
    if px.shape[1] % 2 == 0:
        output = np.zeros((l, nout))
    else:
        output = np.zeros((l, nout+1))
        output[:,-1] = px[:,-1]
    ## approximates actual convolution, since we efficiently implement operations using polyphase filters
    # output[0:2*(l//2),0:nout] = polyphase_filt(px[:,:nout*2:2], px[:,1:nout*2:2], 2)
    # ## performs actual convolution, a bit slower
    # output[0:2*(l//2),0:nout] = decimate_fftconv(px[:,:nout*2:2], px[:,1:nout*2:2], 2)
    ## performs clipping convolution, idea is really high benefits are pretty unlikely
    output[0:2*(l//2),0:nout] = clip_fftconv(px[:,:nout*2:2], px[:,1:nout*2:2], 2)
    output = output.clip(0,np.inf)
    if output.shape[1] >= 2:
        output = sum_pdf_new(output)
    return output / np.sum(output+1e-16,axis=0)

def sum_pdf_fixed_point(x,px,**kwargs):
    ## Calculates pdf of sum of RVs, each column is a pdf of a RV
    ## so px is n_bins x n_RVs
    ## output is summed pdf (length n_bins), associated x region
    ## we compute x regions by focusing on region of pmf that has concentrated probability

    if px.shape[1] == 1:
        return x, px
    l = px.shape[0]
    nout = px.shape[1]//2
    if px.shape[1] % 2 == 0:
        out_x = np.zeros((l, nout))
        out_px = np.zeros((l, nout))
    else:
        out_x = np.zeros((l, nout+1))
        out_px = np.zeros((l, nout+1))
        out_x[:,-1] = x[:,-1]
        out_px[:,-1] = px[:,-1]
    ### Perform convolution and update x region
    out_x[0:2*(l//2),0:nout], out_px[0:2*(l//2),0:nout] = fixed_point_fftconv(x[:,:nout*2:2], px[:,:nout*2:2], 
        x[:,1:nout*2:2], px[:,1:nout*2:2],**kwargs)
    out_px = out_px.clip(0,np.inf)
    if out_px.shape[1] >= 2:
        out_x, out_px = sum_pdf_fixed_point(out_x, out_px,**kwargs)
        
    return out_x, out_px / np.sum(out_px+1e-16,axis=0)

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


def with_x_unit_test(plot=False):
    n_users = 8
    n_bins = 300
    x = np.zeros((n_bins,n_users))
    min_x,max_x = -1*MAX_LATENCY,0
    for i in range(n_users):
        x[:,i] = np.linspace(min_x,max_x,num=n_bins)
    x[:,0] *= 2
    means = min_x+10+(max_x-min_x)*np.random.uniform(size=n_users)
    var=5
    px = np.exp(-np.power((x - means),2) / (2 * np.power(var,2)))
    px[:,-1] = 0
    px[-1,-1] = 1
    # px[-1,-2] = 1
    # px[-1,-3] = 1
    px = px / np.sum(px,axis=0)

    out_x, out_px = sum_pdf_fixed_point(x, px, verbose=True)
    if plot:
        for i in range(n_users):
            plt.plot(x[:,i],px[:,i],label='user {}'.format(i))
        plt.plot(out_x, out_px,label='sum')
        plt.legend()
        plt.savefig('test.pdf')

def with_x_unit_test_realistic(plot=False):
    n_users = 800
    n_bins = LBX_DENSITY
    x = np.zeros((n_bins,n_users))
    min_x,max_x = -1*MAX_LATENCY,0
    for i in range(n_users):
        x[:,i] = np.linspace(min_x,max_x,num=n_bins)
    x[:,0] *= 2

    px = np.zeros((n_bins,n_users))
    for i in range(n_users):
        n_rand_peaks = np.random.randint(1,high=4)
        rand_inds = np.random.randint(0,high=n_bins,size=n_rand_peaks)
        probs = np.random.uniform(size=n_rand_peaks)
        px[rand_inds,i] = probs

    px = px / np.sum(px,axis=0)

    out_x, out_px = sum_pdf_fixed_point(x, px, verbose=True)
    if plot:
        for i in range(n_users):
            plt.plot(x[:,i],px[:,i],label='user {}'.format(i))
        plt.plot(out_x, out_px,label='sum')
        plt.legend()
        plt.savefig('test.pdf')

def old_unit_test():
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
    plt.savefig('test.pdf')


if __name__ == "__main__":
    np.random.seed(31415)
    n_iter = 10
    ts = time.time()
    for i in range(n_iter):
        with_x_unit_test_realistic(plot=(i==n_iter-1))
    print("{} s per iter".format(round((time.time() - ts) / n_iter,2)))





