from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
import torch

import os
import sys
conf_path = os.path.split(__file__)[0]
sys.path.append(os.path.join(conf_path, 'plot'))
import viz2d as viz
import utils as viz_utils


def prepare_data(pipe_data, p=0, m_mask=None, s=1., t=0.):
    im = Image.open(pipe_data['img'][p])
    sz = im.size
        
    kp = pipe_data['kp'][p][pipe_data['m_idx'][pipe_data['m_mask']][:, p]].to('cpu').numpy()
    if not (m_mask is None): kp = kp[m_mask]

    kp_ = (np.round(kp * s + t) - t) / s
    ku, u2k, k2u = np.unique(kp_, return_index=True, return_inverse=True, axis=0)
                
    try: tri = Delaunay(ku)
    except: return None, None, None, None, None 
    t = tri.simplices
    
    avg_l = np.ceil((sz[0] * sz[1] / t.shape[0] * 4 / (3 ** 0.5)) ** 0.5)
    
    l0 = sz[0] / np.ceil(sz[0] / avg_l)
    l1 = sz[1] / np.ceil(sz[1] / avg_l)
    
    q0 = np.arange(0, sz[0] + 0.001, l0)
    q1 = np.arange(0, sz[1] + 0.001, l1)
    
    np.full((q0.shape[0], ), 0.)
    
    b0 = np.stack((q0, np.full((q0.shape[0], ), 0.)))
    b1 = np.stack((q0, np.full((q0.shape[0], ), float(sz[1]))))
    b2 = np.stack((np.full((q1.shape[0], ), 0.), q1))
    b3 = np.stack((np.full((q1.shape[0], ), float(sz[0])), q1))
    b = np.transpose(np.concatenate((b0, b1, b2, b3), axis=1), axes=[1, 0])
    b_ = np.round(b)
    
    kb = np.concatenate((kp_, b_))
    ku, u2k, k2u = np.unique(kb, return_index=True, return_inverse=True, axis=0)
    b_mask = u2k >= kp_.shape[0]

    u2k_list = [[] for i in range(u2k.shape[0])]
    for i in range(kp.shape[0]):
        u2k_list[k2u[i]].append(i)
        
    u2k_list = [np.asarray(i) for i in u2k_list]
    
    try: tri = Delaunay(ku)
    except: return None, None, None, None, None 
    t = tri.simplices
    t_mask = ~np.reshape(b_mask[t.flatten()], [-1, 3]).any(axis=1)
    
    e = np.concatenate((t[~t_mask][:, [0, 1]], t[~t_mask][:, [1, 2]], t[~t_mask][:, [0, 2]]), axis=0)
    e_mask = np.reshape(~b_mask[e.flatten()], [-1, 2]).all(axis=1)
    
    bi = np.unique(np.concatenate((e[e_mask][:, 0], e[e_mask][:, 1]), axis=0))
    bb_mask = np.zeros(ku.shape[0], dtype=bool)
    bb_mask[bi] = 1
    
    b = np.stack((b_mask, bb_mask), axis=1)
    
    vv = np.zeros(ku.shape[0] ** 2, bool)
    e = np.concatenate((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]), axis=0)
    e_mask = np.any(b_mask[e], axis=1)
    e = e[~e_mask]    
    vv[e[:, 0] * ku.shape[0] + e[:, 1]] = 1
    vv[e[:, 1] * ku.shape[0] + e[:, 0]] = 1
    vv = vv.reshape([ku.shape[0], ku.shape[0]])    
    vv = [np.argwhere(vv[i]).squeeze(1) for i in range(vv.shape[0])]
    
    return tri, k2u, u2k_list, b, vv 


def prepare_data_shaped(pipe_data, p=0, m_mask=None, s=1., t=0., diff_i_limit=2):
    im = Image.open(pipe_data['img'][p])
    sz = im.size

    kp_all = pipe_data['kp'][p].to('cpu').numpy()
    kp_all_ = (np.round(kp_all * s + t) - t) / s
    ku_, _, _ = np.unique(kp_all_, return_index=True, return_inverse=True, axis=0)

    kp = pipe_data['kp'][p][pipe_data['m_idx'][pipe_data['m_mask']][:, p]].to('cpu').numpy()
    if not (m_mask is None): kp = kp[m_mask]

    kp_ = (np.round(kp * s + t) - t) / s
    ku, u2k, k2u = np.unique(kp_, return_index=True, return_inverse=True, axis=0)

    ku_aux, u2k_aux, k2u_aux = np.unique(np.concatenate((ku, ku_)), return_index=True, return_inverse=True, axis=0)
    aux_mask = np.zeros(ku_aux.shape[0], dtype=bool)
    aux_mask[k2u_aux[:ku.shape[0]]] = True    
    kp_out_ = ku_aux[~aux_mask]
                    
    avg_l = np.ceil((sz[0] * sz[1] / ku.shape[0]) ** 0.5)
    
    l0 = sz[0] / np.ceil(sz[0] / avg_l)
    l1 = sz[1] / np.ceil(sz[1] / avg_l)
    
    q0 = np.arange(0, sz[0] + 0.001, l0)
    q1 = np.arange(0, sz[1] + 0.001, l1)
    
    np.full((q0.shape[0], ), 0.)
    
    b0 = np.stack((q0, np.full((q0.shape[0], ), 0.)))
    b1 = np.stack((q0, np.full((q0.shape[0], ), float(sz[1]))))
    b2 = np.stack((np.full((q1.shape[0], ), 0.), q1))
    b3 = np.stack((np.full((q1.shape[0], ), float(sz[0])), q1))
    b = np.transpose(np.concatenate((b0, b1, b2, b3), axis=1), axes=[1, 0])
    b_ = np.round(b)

    kb_tmp = np.concatenate((kp_, kp_out_, b_))
    ku_tmp, u2k_tmp, k2u_tmp = np.unique(kb_tmp, return_index=True, return_inverse=True, axis=0)

    try: tri = Delaunay(ku_tmp)
    except: return None, None, None, None, None 
    t = tri.simplices
    
    diff_i = 1
    diff_mask = np.zeros(ku_tmp.shape[0], dtype=int)
    diff_mask[u2k_tmp >= kp_.shape[0] + kp_out_.shape[0]] = diff_i
    diff_in_set = np.all(u2k_tmp[t] >= kp_.shape[0], axis=1)

    while True:
        border_tri_keep = diff_in_set & np.any(diff_mask[k2u_tmp[u2k_tmp[t]]] > 0, axis=1)
        to_keep = u2k_tmp[t[border_tri_keep].flatten()]
        idx_to_keep = to_keep[(to_keep >= kp_.shape[0]) & (diff_mask[k2u_tmp[to_keep]] == 0)]

        if idx_to_keep.shape[0] == 0: break

        diff_i += 1
        diff_mask[k2u_tmp[idx_to_keep]] = diff_i
        
        if diff_i == diff_i_limit: break
        
        
    # colors = ['g', 'b', 'r', 'c', 'm', 'y', 'w', 'k']

    # fig0, ax0 = plt.subplots(1)
    # im = Image.open(pipe_data['img'][p])
    # ax0.imshow(im)
    # ax0.set_axis_off()
    
    # tpt = tri.points
    # ax0.triplot(tpt[:, 0], tpt[:,1], t, lw=0.2)

    # li = 0
    # q = kp_
    # ax0.plot(q[:, 0], q[:, 1], '.', color=colors[abs(li) % len(colors)], lw=0.2, markersize=0.5)

    # li = 1
    # q = b_[u2k_tmp[diff_mask == li] - kp_.shape[0] - kp_out_.shape[0]] 
    # ax0.plot(q[:, 0], q[:, 1], '.', color=colors[abs(li) % len(colors)], lw=0.2, markersize=0.5)
              
    # for li in range(2, diff_i + 1):
    #     q = kp_out_[u2k_tmp[diff_mask == li] - kp_.shape[0]]
    #     ax0.plot(q[:, 0], q[:, 1], '.', color=colors[abs(li) % len(colors)], lw=0.2, markersize=0.5)


###

    # diff_i = 1
    # diff_mask_ = np.zeros(ku_tmp.shape[0], dtype=int)
    # diff_mask_[u2k_tmp < kp_.shape[0]] = diff_i
    # diff_in_set_ = np.all(u2k_tmp[t] < kp_.shape[0] + kp_out_.shape[0], axis=1)

    # while True:
    #     border_tri_keep = diff_in_set_ & np.any(diff_mask_[k2u_tmp[u2k_tmp[t]]] > 0, axis=1)
    #     to_keep = u2k_tmp[t[border_tri_keep].flatten()]
    #     idx_to_keep = to_keep[(to_keep >= kp_.shape[0]) & (to_keep < kp_.shape[0] + kp_out_.shape[0]) & (diff_mask_[k2u_tmp[to_keep]] == 0)]

    #     if idx_to_keep.shape[0] == 0: break

    #     diff_i += 1
    #     diff_mask_[k2u_tmp[idx_to_keep]] = diff_i
        
        
    # colors = ['g', 'b', 'r', 'c', 'm', 'y', 'w', 'k']

    # fig0, ax0 = plt.subplots(1)
    # im = Image.open(pipe_data['img'][p])
    # ax0.imshow(im)
    # ax0.set_axis_off()
    
    # tpt = tri.points
    # ax0.triplot(tpt[:, 0], tpt[:,1], t, lw=0.2)

    # li = 0
    # q = kp_
    # ax0.plot(q[:, 0], q[:, 1], '.', color=colors[abs(li) % len(colors)], lw=0.2, markersize=0.5)

    # li = 1
    # q = b_ 
    # ax0.plot(q[:, 0], q[:, 1], '.', color=colors[abs(li) % len(colors)], lw=0.2, markersize=0.5)
              
    # for li in range(2, diff_i + 1):
    #     q = kp_out_[u2k_tmp[diff_mask_ == li] - kp_.shape[0]]
    #     ax0.plot(q[:, 0], q[:, 1], '.', color=colors[abs(li) % len(colors)], lw=0.2, markersize=0.5)

###

    kp_out_to_keep = kp_out_[u2k_tmp[(diff_mask >= 2) & (diff_mask <= diff_i_limit)] - kp_.shape[0]]
    b_new_ = np.concatenate((b_, kp_out_to_keep))
    
    kb = np.concatenate((kp_, b_new_))
    ku, u2k, k2u = np.unique(kb, return_index=True, return_inverse=True, axis=0)
    b_mask = u2k >= kp_.shape[0]

    u2k_list = [[] for i in range(u2k.shape[0])]
    for i in range(kp.shape[0]):
        u2k_list[k2u[i]].append(i)
        
    u2k_list = [np.asarray(i) for i in u2k_list]
    
    try: tri = Delaunay(ku)
    except: return None, None, None, None, None 
    t = tri.simplices
    t_mask = ~np.reshape(b_mask[t.flatten()], [-1, 3]).any(axis=1)
    
    e = np.concatenate((t[~t_mask][:, [0, 1]], t[~t_mask][:, [1, 2]], t[~t_mask][:, [0, 2]]), axis=0)
    e_mask = np.reshape(~b_mask[e.flatten()], [-1, 2]).all(axis=1)
    
    bi = np.unique(np.concatenate((e[e_mask][:, 0], e[e_mask][:, 1]), axis=0))
    bb_mask = np.zeros(ku.shape[0], dtype=bool)
    bb_mask[bi] = 1
    
    b = np.stack((b_mask, bb_mask), axis=1)
    
    vv = np.zeros(ku.shape[0] ** 2, bool)
    e = np.concatenate((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]), axis=0)
    e_mask = np.any(b_mask[e], axis=1)
    e = e[~e_mask]    
    vv[e[:, 0] * ku.shape[0] + e[:, 1]] = 1
    vv[e[:, 1] * ku.shape[0] + e[:, 0]] = 1
    vv = vv.reshape([ku.shape[0], ku.shape[0]])    
    vv = [np.argwhere(vv[i]).squeeze(1) for i in range(vv.shape[0])]
    
    return tri, k2u, u2k_list, b, vv 


def check_in_tri(tri, tri_pt, query_pt):
    tri_x = tri_pt[: ,0][tri]
    tri_y = tri_pt[:, 1][tri]    

    mx = np.repeat(np.min(tri_x, axis=1)[np.newaxis, :], query_pt.shape[0], axis=0)
    my = np.repeat(np.min(tri_y, axis=1)[np.newaxis, :], query_pt.shape[0], axis=0)

    Mx = np.repeat(np.max(tri_x, axis=1)[np.newaxis, :], query_pt.shape[0], axis=0)
    My = np.repeat(np.max(tri_y, axis=1)[np.newaxis, :], query_pt.shape[0], axis=0)

    x = np.repeat(query_pt[:, 0][:, np.newaxis], tri.shape[0], axis=1)
    y = np.repeat(query_pt[:, 1][:, np.newaxis], tri.shape[0], axis=1)

    check0 = (x >= mx) & (x <= Mx) & (y >= my) & (y <= My)

    if not np.any(check0): return np.full(query_pt.shape[0], -1, dtype=int)

    tri_idx = np.any(check0, axis=0)
    query_idx = np.any(check0, axis=1)
    
    tri_ = tri[tri_idx]
    query_pt_ = query_pt[query_idx]
    
    tri_iidx = np.argwhere(tri_idx)
    query_iidx = np.argwhere(query_idx)

    x1 = tri_pt[:, 0][tri_[:, 0]]
    y1 = tri_pt[:, 1][tri_[:, 0]]
    x0 = tri_pt[:, 0][tri_[:, 1]]
    y0 = tri_pt[:, 1][tri_[:, 1]]
    side1 = np.stack((y1-y0, -x1+x0, y0*(x1-x0)-x0*(y1-y0))).transpose(1, 0)
    # xy = np.concatenate((tri_pt[tri_[:, 2]], np.ones((tri_.shape[0], 1))), axis=1)
    # sign1 = np.sign(np.sum(side1 * xy, axis=1))
    
    x1 = tri_pt[:, 0][tri_[:, 1]]
    y1 = tri_pt[:, 1][tri_[:, 1]]
    x0 = tri_pt[:, 0][tri_[:, 2]]
    y0 = tri_pt[:, 1][tri_[:, 2]]
    side2 = np.stack((y1-y0, -x1+x0, y0*(x1-x0)-x0*(y1-y0))).transpose(1, 0)
    # xy = np.concatenate((tri_pt[tri_[:, 0]], np.ones((tri_.shape[0], 1))), axis=1)
    # sign2 = np.sign(np.sum(side2 * xy, axis=1))

    x1 = tri_pt[:, 0][tri_[:, 2]]
    y1 = tri_pt[:, 1][tri_[:, 2]]
    x0 = tri_pt[:, 0][tri_[:, 0]]
    y0 = tri_pt[:, 1][tri_[:, 0]]
    side3 = np.stack((y1-y0, -x1+x0, y0*(x1-x0)-x0*(y1-y0))).transpose(1, 0)
    # xy = np.concatenate((tri_pt[tri_[:, 1]], np.ones((tri_.shape[0], 1))), axis=1)
    # sign3 = np.sign(np.sum(side3 * xy, axis=1))
    
    check0_ = check0[query_idx][:, tri_idx]
    to_check = np.argwhere(check0_)

    to_check_query_pt = np.concatenate((query_pt_[to_check[:, 0]], np.ones((to_check.shape[0],1))), axis=1)
    # sign1, sign2, sign3 always 1 since Delaunay triangle are ordered clockwise
    check1 = (np.sum(side1[to_check[:, 1]] * to_check_query_pt, axis=1) > 0) & (np.sum(side2[to_check[:, 1]] * to_check_query_pt, axis=1) > 0) & (np.sum(side3[to_check[:, 1]] * to_check_query_pt, axis=1) > 0)

    if not np.any(check1): return np.full(query_pt.shape[0], -1, dtype=int)
    
    check_tri = np.full(query_pt.shape[0], -1, dtype=int)            
    check_tri[query_iidx[to_check[:, 0][check1]]] = tri_iidx[to_check[:, 1][check1]]

    return check_tri


def is_in_tri(tri, query_pt, b=None, max_tri=20000, max_query_pt=20000):
    if b is None:
        masked_tri = tri.simplices
    else:
        mask_tri = np.all(~b[tri.simplices], axis=1)
        masked_tri = tri.simplices[mask_tri]
        
    in_tri = np.full(query_pt.shape[0], -1, dtype=int)
    for i in range(0, query_pt.shape[0], max_query_pt):
        current_query_pt = query_pt[i:min(query_pt.shape[0], i + max_query_pt)]
        tmp = np.full(current_query_pt.shape[0], -1, dtype=int)
        for j in range(0, masked_tri.shape[0], max_tri):
            current_tri = masked_tri[j:min(masked_tri.shape[0], j + max_tri)]
            aux = check_in_tri(current_tri, tri.points, current_query_pt)
            tmp[aux > -1] = aux[aux > -1] + j
        in_tri[i:min(query_pt.shape[0], i + max_query_pt)] = tmp                        

    if not (b is None):
        aux = np.arange(tri.simplices.shape[0])[mask_tri]
        in_tri[in_tri > -1] = aux[in_tri[in_tri > -1]]

    return in_tri


def plot_tri(pipe_data, p, tri, k2u, b, title=None):
    fig, ax = plt.subplots(1)

    im = Image.open(pipe_data['img'][p])
    ax.imshow(im)
    ax.set_axis_off()
    
    t = tri.simplices
    t_mask = ~np.reshape(b[:,0][t.flatten()], [-1, 3]).any(axis=1)    
    
    ku = tri.points

    ax.triplot(ku[:, 0], ku[:,1], t[t_mask], lw=0.2)
    ax.triplot(ku[:, 0], ku[:,1], t[~t_mask], lw=0.2, color='r')
    ax.plot(ku[b[:, 1], 0], ku[b[:, 1], 1], 'g.', markersize=0.5)
    
    if not (title is None): ax.set_title(title) 
    
    return fig, ax
        
    
def plot_matches(pipe_data, m_mask, title=None):
    if m_mask is None: m_mask = np.zeros(pipe_data['m_mask'].shape[0], dtype=int)
        
    colors = ['g', 'b', 'r', 'c', 'm', 'y', 'w', 'k']
    l = np.min(m_mask)

    fig0, ax0 = plt.subplots(1)
    im = Image.open(pipe_data['img'][0])
    ax0.imshow(im)
    ax0.set_axis_off()
    
    if not (title is None): ax0.set_title(title)
        
    for li in range(0, l - 1, -1):
        c_mask = (m_mask == li)
        
        q0 = pipe_data['kp'][0][pipe_data['m_idx'][pipe_data['m_mask']][:, 0]].to('cpu').numpy()
        q1 = pipe_data['kp'][1][pipe_data['m_idx'][pipe_data['m_mask']][:, 1]].to('cpu').numpy()
        x = np.stack((q0[:, 0], q1[:, 0]))[:, c_mask]
        y = np.stack((q0[:, 1], q1[:, 1]))[:, c_mask]
        ax0.plot(x, y, '-', color=colors[abs(li) % len(colors)], lw=0.2)

    fig1, ax1 = plt.subplots(1)
    im = Image.open(pipe_data['img'][1])
    ax1.imshow(im)
    ax1.set_axis_off()
    
    if not (title is None): ax1.set_title(title)

    for li in range(0, l - 1, -1):
        c_mask = (m_mask == li)
        
        q0 = pipe_data['kp'][0][pipe_data['m_idx'][pipe_data['m_mask']][:, 0]].to('cpu').numpy()
        q1 = pipe_data['kp'][1][pipe_data['m_idx'][pipe_data['m_mask']][:, 1]].to('cpu').numpy()
        x = np.stack((q0[:, 0], q1[:, 0]))[:, c_mask]
        y = np.stack((q0[:, 1], q1[:, 1]))[:, c_mask]
        ax1.plot(x, y, '-', color=colors[abs(li) % len(colors)], lw=0.2)
    
    return [fig0, fig1], [ax0, ax1]


def in_tri_show(pipe_data, tri, in_tri, to_check_tri_pt, p=0):
    fig, ax = plt.subplots(1)
    im = Image.open(pipe_data['img'][p])
    ax.imshow(im)
    ax.set_axis_off()        

    no_tri_pt = to_check_tri_pt[in_tri == -1]
    ax.plot(no_tri_pt[:, 0], no_tri_pt[:, 1], '.', markersize=0.5)
    
    with_tri = to_check_tri_pt[in_tri > -1]
    related_tri = in_tri[in_tri > -1]
        
    related_tri_mask = np.zeros(np.max(related_tri) + 1, dtype=bool)
    clr = ['b', 'g', 'r', 'c', 'm', 'y',  'w', 'k']
    for ri in range(with_tri.shape[0]):
        in_tri_pt = with_tri[ri]
        rel_tri = tri.simplices[related_tri[ri]]
        color = clr[related_tri[ri] % len(clr)]
        tt_ = np.append(rel_tri, rel_tri[0])

        if not related_tri_mask[related_tri[ri]]:
            tx = tri.points[tt_, 0]
            ty = tri.points[tt_, 1]

            mx = np.mean(tx[:3])
            my = np.mean(ty[:3])

            qx = mx - tx
            qy = my - ty

            tx = tx + qx * 0.1
            ty = ty + qy * 0.1

            ax.plot(tx, ty, color=color, lw=0.5)
            
            related_tri_mask[related_tri[ri]] = True
        ax.plot(in_tri_pt[0], in_tri_pt[1], '+', markersize=0.5, color=color)

    return fig, ax


def dtm(pipe_data, show_in_progress=False, full_dtm=True, st=[1., 0.], prepare_data=prepare_data_shaped):
    if show_in_progress: plot_matches(pipe_data, None, title='DTM - input')
    
    mask = dtm1(pipe_data, show_in_progress=show_in_progress, st=st, prepare_data=prepare_data)

    if full_dtm:
        if not (mask is None):                
            return dtm2(pipe_data, mask, show_in_progress=show_in_progress, st=st, prepare_data=prepare_data)

    return mask

    
def dtm1(pipe_data, show_in_progress=False, st=[1., 0.], prepare_data=prepare_data_shaped):
    mask = None
    it = 0
    while True:    
        cmask = None if (mask is None) else (mask == 0)
        
        tri0, k2u0, u2k0, b0, e0 = prepare_data(pipe_data, p=0, m_mask=cmask, s=st[0], t=st[1])
        tri1, k2u1, u2k1, b1, e1 = prepare_data(pipe_data, p=1, m_mask=cmask, s=st[0], t=st[1])

        if (tri0 is None) or (tri1 is None): return mask        
        
        if show_in_progress:
            plot_tri(pipe_data, 0, tri0, k2u0, b0, title='DTM1 - iter. ' + str(it))
            plot_tri(pipe_data, 1, tri1, k2u1, b1, title='DTM1 - iter. ' + str(it))
        
        mm = pipe_data['m_idx'][pipe_data['m_mask']]
        mv = pipe_data['m_val'][pipe_data['m_mask']]
    
        if not (mask is None):
            mm = mm[cmask]
            mv = mv[cmask]
        
        l = mm.shape[0]
        
        m1 = np.zeros((l, l), bool)
        for i in range(l):
            aux = [u2k0[j] for j in e0[k2u0[i]]]
            if len(aux): m1[i, np.concatenate(aux)] = 1
        
        m2 = np.zeros((l, l), bool)
        for i in range(l):
            aux = [u2k1[j] for j in e1[k2u1[i]]]
            if len(aux): m2[i, np.concatenate(aux)] = 1
            
        qi_r1 = m1 & m2
        qd_r1 = np.logical_xor(m1, m2)    
        
        matches_r1 = np.sum(qi_r1, axis=1)  
        
        iaux = np.argsort(-matches_r1)
        iidx = np.argsort(mv.to('cpu').numpy()[iaux], kind='stable')
        iidx = iaux[iidx]
        jidx = np.argsort(iidx)
        
        t = np.ones(iidx.shape[0], dtype=bool)
        for j in range(t.shape[0]):
            if t[j]: t[jidx[qd_r1[iidx[j], :]]] = False   
        t = t[jidx]
        m_mask = np.zeros(iidx.shape[0], dtype=bool)
        for j in range(m_mask.shape[0]):
            if t[j]: m_mask[qi_r1[j, :]] = True
                
        if mask is None:
            it = 1
            mask = (~m_mask).astype(int)
        else:
            it += 1
            mask[mask == 0] = (~m_mask).astype(int) * it
            
        # if it > 1:
        #     to_check_tri_pt0 = pipe_data['kp'][0][pipe_data['m_idx'][pipe_data['m_mask']][:, 0]].to('cpu').numpy()[mask == 1]
        #     in_tri0 = is_in_tri(tri0, to_check_tri_pt0, b=b0[:, 0])
    
        #     in_tri_show(pipe_data, tri0, in_tri0, to_check_tri_pt0, p=0)
    
        #     to_check_tri_pt1 = pipe_data['kp'][1][pipe_data['m_idx'][pipe_data['m_mask']][:, 1]].to('cpu').numpy()[mask == 1]
        #     in_tri1 = is_in_tri(tri1, to_check_tri_pt1, b=b1[:, 0])
    
        #     in_tri_show(pipe_data, tri1, in_tri1, to_check_tri_pt1, p=1)
                    
        if show_in_progress: plot_matches(pipe_data, mask, title='DTM1 - iter. ' + str(it))
                
        if np.all(m_mask): break
    
    return mask


def dtm2(pipe_data, mask, show_in_progress=False, st=[1., 0.], prepare_data=prepare_data_shaped):
    
    l = np.max(np.unique(mask))
    for li in range(l,0,-1):
        cmask = (mask <= 0)
        
        tri0, k2u0, u2k0, b0, e0 = prepare_data(pipe_data, p=0, m_mask=cmask, s=st[0], t=st[1])
        tri1, k2u1, u2k1, b1, e1 = prepare_data(pipe_data, p=1, m_mask=cmask, s=st[0], t=st[1])
        
        if (tri0 is None) or  (tri1 is None): return mask        
        
        if show_in_progress and (li != l):
            plot_tri(pipe_data, 0, tri0, k2u0, b0, title='DTM2 - iter. ' + str(li - 1))
            plot_tri(pipe_data, 1, tri1, k2u1, b1, title='DTM2 - iter. ' + str(li - 1))
        
        to_check = (mask == li)
    
        to_check_tri_pt0 = pipe_data['kp'][0][pipe_data['m_idx'][pipe_data['m_mask']][:, 0]].to('cpu').numpy()[to_check]
        to_check_tri_pt1 = pipe_data['kp'][1][pipe_data['m_idx'][pipe_data['m_mask']][:, 1]].to('cpu').numpy()[to_check]
    
        m0 = pipe_data['kp'][0][pipe_data['m_idx'][pipe_data['m_mask']][:, 0]].to('cpu').numpy()[cmask]
        m1 = pipe_data['kp'][1][pipe_data['m_idx'][pipe_data['m_mask']][:, 1]].to('cpu').numpy()[cmask]
    
        to_check_good = np.zeros(np.sum(to_check), dtype=bool)
    
        in_tri0 = is_in_tri(tri0, to_check_tri_pt0)
        in_tri1 = is_in_tri(tri1, to_check_tri_pt1)
    
        for j in range(in_tri0.shape[0]):
            if (in_tri0[j] == -1) or ((in_tri1[j] == -1)):
                continue
            
            aux_tri = tri0.simplices[in_tri0[j]]
            tri_idx = aux_tri[~b0[:, 0][aux_tri]]
    
            if (len(tri_idx) < 3):
                if len(tri_idx) < 1:
                    to_check_good[j] = False
                    continue            
                            
                tri_pts = tri0.points[tri_idx]
                
                tidx = tri_idx[np.argmin(np.sum((tri_pts - to_check_tri_pt0[j]) ** 2, axis=1))]
                list_pts = [u2k0[q] for q in e0[tidx]]
                if len(list_pts) == 0:                
                    to_check_good[j] = False
                    continue
                list_pts = np.concatenate(list_pts)
                reproj_pt0 = to_check_tri_pt1[j] + m0[list_pts] - m1[list_pts]
    
                q_tri = check_in_tri(aux_tri[np.newaxis, :], tri0.points, reproj_pt0)
                if np.any(q_tri > -1): to_check_good[j] = True
                continue
            
            q0 = m1[u2k0[aux_tri[0]]]
            q1 = m1[u2k0[aux_tri[1]]]
            q2 = m1[u2k0[aux_tri[2]]]
            
            q_tri = np.zeros((q0.shape[0] * q1.shape[0] * q2.shape[0], 3), dtype=int)
            ii = 0
            for j0 in range(q0.shape[0]):
                for j1 in range(q1.shape[0]):
                    for j2 in range(q2.shape[0]):
                        q_tri[ii, 0] = j0    
                        q_tri[ii, 1] = j1 + q0.shape[0]    
                        q_tri[ii, 2] = j2 + q0.shape[0] + q1.shape[0]   
                        ii += 1
            q_tri = check_in_tri(q_tri, np.concatenate((q0, q1, q2), axis=0), to_check_tri_pt1[j][np.newaxis, :])
            if (q_tri[0] > -1): to_check_good[j] = True
    
        for j in range(in_tri1.shape[0]):
            if not to_check_good[j]:
                continue
            
            aux_tri = tri1.simplices[in_tri1[j]]
            tri_idx = aux_tri[~b1[:, 0][aux_tri]]
    
            if (len(tri_idx) < 3):
                if len(tri_idx) < 1:
                    to_check_good[j] = False
                    continue
                
                tri_pts = tri1.points[tri_idx]
                tidx = tri_idx[np.argmin(np.sum((tri_pts - to_check_tri_pt1[j]) ** 2, axis=1))]
                list_pts = [u2k1[q] for q in e1[tidx]]
                if len(list_pts) == 0:                
                    to_check_good[j] = False
                    continue
                list_pts = np.concatenate(list_pts)
                reproj_pt1 = to_check_tri_pt0[j] + m1[list_pts] - m0[list_pts]
    
                q_tri = check_in_tri(aux_tri[np.newaxis, :], tri1.points, reproj_pt1)
                if not np.any(q_tri > -1): to_check_good[j] = False
                continue
            
            q0 = m0[u2k1[aux_tri[0]]]
            q1 = m0[u2k1[aux_tri[1]]]
            q2 = m0[u2k1[aux_tri[2]]]
            
            q_tri = np.zeros((q0.shape[0] * q1.shape[0] * q2.shape[0], 3), dtype=int)
            ii = 0
            for j0 in range(q0.shape[0]):
                for j1 in range(q1.shape[0]):
                    for j2 in range(q2.shape[0]):
                        q_tri[ii, 0] = j0    
                        q_tri[ii, 1] = j1 + q0.shape[0]    
                        q_tri[ii, 2] = j2 + q0.shape[0] + q1.shape[0]   
                        ii += 1
            q_tri = check_in_tri(q_tri, np.concatenate((q0, q1, q2), axis=0), to_check_tri_pt0[j][np.newaxis, :])
            if not (q_tri[0] > -1): to_check_good[j] = False
            
        checked_mask = np.full(np.sum(to_check), li, dtype=int)
        checked_mask[to_check_good] = -li
        mask[to_check] = checked_mask
            
        if show_in_progress: plot_matches(pipe_data, mask, title='DTM2 - iter. ' + str(li - 1))

    if show_in_progress:        
        cmask = (mask <= 0)
        
        tri0, k2u0, u2k0, b0, e0 = prepare_data(pipe_data, p=0, m_mask=cmask, s=st[0], t=st[1])
        tri1, k2u1, u2k1, b1, e1 = prepare_data(pipe_data, p=1, m_mask=cmask, s=st[0], t=st[1])        
        
        plot_tri(pipe_data, 0, tri0, k2u0, b0, title='DTM2 - end')
        plot_tri(pipe_data, 1, tri1, k2u1, b1, title='DTM2 - end')

    return mask


def blob_matching(pt1, pt2, desc1, desc2,
                  pf=-10,  # f = 10 with union
                  pn=3,    # f' = 5
                  ps=16,   # to = 10
                  use_stats=True,
                  out_idx=10,
                  distance='L2',
                  ss=1024, # split size
                  same_order=True,    
                  device='cpu', # preferred to avoid OOM
        ):

    if pt1.device != device: pt1 = pt1.to(device)
    if pt2.device != device: pt2 = pt2.to(device)
    if desc1.device != device: desc1 = desc1.to(device)
    if desc2.device != device: desc2 = desc2.to(device)
        
    desc1_blk = torch.split(desc1, ss, dim=0)
    desc2_blk = torch.split(desc2, ss, dim=0)

    m = torch.zeros((pt1.shape[0], pt2.shape[0]), dtype=torch.float32, device=device)
            
    pmn = 2
    if distance == 'L1': pmn = 1
    
    for i in torch.arange(0, len(desc1_blk)):
        for j in torch.arange(0, len(desc2_blk)):
            v1 = desc1_blk[i]
            v2 = desc2_blk[j]

            m[i * ss:i * ss + v1.shape[0], j * ss:j * ss + v2.shape[0]] = torch.cdist(v1.unsqueeze(0), v2.unsqueeze(0), p=pmn)

    m[m==0] = 1.0e-18        
    
    s = m.shape
    pf = pf if pf else torch.inf
    sign_pf = np.sign(pf).item()
    pf = abs(pf)

    pf1 = min(pf, s[0])
    pf2 = min(pf, s[1])
    
    mr = torch.sort(m, dim=0)[0][pf1, :].unsqueeze(0).repeat(s[0], 1)
    mc = torch.sort(m, dim=1)[0][:, pf2].unsqueeze(1).repeat(1, s[1])
    
    mm = m.clone()
    if sign_pf > 0:
        mm[(mm > mr) & (mm > mc)] = torch.inf
    else:
        mm[(mm > mr) | (mm > mc)] = torch.inf
    
    t, idx = torch.sort(mm.flatten())
    i = idx // s[1]
    j = idx % s[1]

    kt = torch.isfinite(t)
    idx = idx[kt]
    t = t[kt].to('cpu').numpy()
    i = i[kt].to('cpu').numpy()
    j = j[kt].to('cpu').numpy()
    
    r = np.zeros(s[0], dtype=int)
    c = np.zeros(s[1], dtype=int)
    l = min(s[0], s[1])
    p = np.zeros((l * pn, 2), dtype=int)
    v = np.zeros(l * pn)
            
    kc = 0
    for k in range(idx.shape[0]):
        if (r[i[k]] < pn) and (c[j[k]] < pn):
            r[i[k]] += 1
            c[j[k]] += 1
            p[kc, 0] = i[k]
            p[kc, 1] = j[k]
            v[kc] = t[k]
            kc += 1
        if kc >= l * pn: break

    p = torch.tensor(p[:kc], device=device)
    v = torch.tensor(v[:kc], dtype=torch.float, device=device)

    if not use_stats:
        stat_a = torch.zeros((pt1.shape[0], pt1.shape[0]), device=device, dtype=bool)
        stat_b = torch.zeros((pt2.shape[0], pt2.shape[0]), device=device, dtype=bool)
    else:
        stat_a = torch.zeros((pt1.shape[0], pt1.shape[0]), dtype=torch.float32, device=device)            
        pt1_blk = torch.split(pt1, ss, dim=0)
        
        for i in torch.arange(0, len(pt1_blk)):
            for j in torch.arange(i, len(pt1_blk)):
                v1 = pt1_blk[i]
                v2 = pt1_blk[j]
                    
                stat_a[i * ss:i * ss + v1.shape[0], j * ss:j * ss + v2.shape[0]] = ((v1[:, 0].unsqueeze(-1) - v2[:, 0].unsqueeze(0)) ** 2 + (v1[:, 1].unsqueeze(-1) - v2[:, 1].unsqueeze(0)) ** 2) ** 0.5
                stat_a[j * ss:j * ss + v2.shape[0], i * ss:i * ss + v1.shape[0]] = stat_a[i * ss:i * ss + v1.shape[0], j * ss:j * ss + v2.shape[0]].T.clone()

        stat_a = stat_a <= ps          

        stat_b = torch.zeros((pt2.shape[0], pt2.shape[0]), dtype=torch.float32, device=device)            
        pt2_blk = torch.split(pt2, ss, dim=0)
        
        for i in torch.arange(0, len(pt2_blk)):
            for j in torch.arange(i, len(pt2_blk)):
                v1 = pt2_blk[i]
                v2 = pt2_blk[j]
                    
                stat_b[i * ss:i * ss + v1.shape[0], j * ss:j * ss + v2.shape[0]] = ((v1[:, 0].unsqueeze(-1) - v2[:, 0].unsqueeze(0)) ** 2 + (v1[:, 1].unsqueeze(-1) - v2[:, 1].unsqueeze(0)) ** 2) ** 0.5
                stat_b[j * ss:j * ss + v2.shape[0], i * ss:i * ss + v1.shape[0]] = stat_b[i * ss:i * ss + v1.shape[0], j * ss:j * ss + v2.shape[0]].T.clone()

        stat_b = stat_b <= ps   

    l = p.shape[0]
    pp = torch.zeros((l, 15), device=device)

    vr = torch.zeros(l, device=device)
    vc = torch.zeros(l, device=device)

    vr_ = torch.zeros(l, device=device)
    vc_ = torch.zeros(l, device=device)

    for kk in range(0, l, ss):
        kk_ = min(kk + ss, l)

        aux_r = m[p[kk:kk_, 0]].clone()
        aux_r[aux_r < v[kk:kk_].unsqueeze(-1)] = torch.inf

        s_aux = aux_r.shape[0]
        aux_r = aux_r.flatten()
        aux_r[torch.arange(0, s_aux, device=device) * m.shape[1] + p[kk:kk_, 1]] = torch.inf
        aux_r = aux_r.reshape(-1, m.shape[1])

        mask_r = stat_b.permute(1, 0)[p[kk:kk_, 1]].permute(0, 1)
        aux_r[mask_r] = torch.inf            

        vr[kk:kk_] = torch.min(aux_r, dim=1)[0]

        aux_c = m.permute(1, 0)[p[kk:kk_, 1]]
        aux_c[aux_c < v[kk:kk_].unsqueeze(-1)] = torch.inf

        s_aux = aux_c.shape[0]
        aux_c = aux_c.flatten()
        aux_c[torch.arange(0, s_aux, device=device) * m.shape[0] + p[kk:kk_, 0]] = torch.inf
        aux_c = aux_c.reshape(-1, m.shape[0])

        mask_c = stat_a[p[kk:kk_, 0]]
        aux_c[mask_c] = torch.inf            

        vc[kk:kk_] = torch.min(aux_c, dim=1)[0]
        
        aux_r = m[p[kk:kk_, 0]].clone()

        s_aux = aux_r.shape[0]
        aux_r = aux_r.flatten()
        aux_r[torch.arange(0, s_aux, device=device) * m.shape[1] + p[kk:kk_, 1]] = torch.inf
        aux_r = aux_r.reshape(-1, m.shape[1])

        mask_r = stat_b.permute(1, 0)[p[kk:kk_, 1]].permute(0, 1)
        aux_r[mask_r] = torch.inf            

        vr_[kk:kk_] = torch.min(aux_r, dim=1)[0]

        aux_c = m.permute(1, 0)[p[kk:kk_, 1]]

        s_aux = aux_c.shape[0]
        aux_c = aux_c.flatten()
        aux_c[torch.arange(0, s_aux, device=device) * m.shape[0] + p[kk:kk_, 0]] = torch.inf
        aux_c = aux_c.reshape(-1, m.shape[0])

        mask_c = stat_a[p[kk:kk_, 0]]
        aux_c[mask_c] = torch.inf            

        vc_[kk:kk_] = torch.min(aux_c, dim=1)[0]
        

    vr[torch.isinf(vr)] = v[torch.isinf(vr)]
    vc[torch.isinf(vc)] = v[torch.isinf(vc)]

    vr_[torch.isinf(vr_)] = v[torch.isinf(vr_)]
    vc_[torch.isinf(vc_)] = v[torch.isinf(vc_)]

    pp[:, 0] = 2 * v / (vr + vc)                
    pp[:, 1] = torch.minimum(v / vr, v / vc)              
    pp[:, 2] = torch.maximum(v / vr, v / vc)                
    pp[:, 3] = v / vr                
    pp[:, 4] = v / vc                

    pp[:, 5] = (2 * v) / (2 * v + vr + vc)                
    pp[:, 6] = torch.minimum(v / (v + vr), v / (v + vc))                
    pp[:, 7] = torch.maximum(v / (v + vr), v / (v + vc))  
    pp[:, 8] = v / (v + vr)                
    pp[:, 9] = v / (v + vc)           

    pp[:, 10] = (2 * v) / (2 * v + vr_ + vc_)                
    pp[:, 11] = torch.minimum(v / (v + vr_), v / (v + vc_))                
    pp[:, 12] = torch.maximum(v / (v + vr_), v / (v + vc_))  
    pp[:, 13] = v / (v + vr_)                
    pp[:, 14] = v / (v + vc_)   

    if same_order:
        pp[:, 10] = pp[:, 10] / (1 - pp[:, 10])                
        pp[:, 11] = pp[:, 11] / (1 - pp[:, 11])                
        pp[:, 12] = pp[:, 12] / (1 - pp[:, 12])                
        pp[:, 13] = pp[:, 13] / (1 - pp[:, 13])                
        pp[:, 14] = pp[:, 14] / (1 - pp[:, 14])  

    # for k in range(l):
    #     v = m[p[k, 0], p[k, 1]]
        
    #     aux_r = m[p[k, 0]].clone()
    #     aux_r[aux_r < v] = torch.inf
    #     aux_r[p[k, 1]] = torch.inf
        
    #     mask_r = stat_b[p[k, 1]]
    #     aux_r[mask_r] = torch.inf
        
    #     vr = aux_r.min()

    #     aux_c = m[:, p[k, 1]].clone()
    #     aux_c[aux_c < v] = torch.inf
    #     aux_c[p[k, 0]] = torch.inf
        
    #     mask_c = stat_a[p[k, 0]]
    #     aux_c[mask_c] = torch.inf
        
    #     vc = aux_c.min()
        
    #     if torch.isinf(vr): vr = v
    #     if torch.isinf(vc): vc = v

    #     pp[k, 0] = 2 * v / (vr + vc)                
    #     pp[k, 1] = min(v / vr, v / vc)                
    #     pp[k, 2] = max(v / vr, v / vc)                
    #     pp[k, 3] = v / vr                
    #     pp[k, 4] = v / vc                

    #     pp[k, 5] = (2 * v) / (2 * v + vr + vc)                
    #     pp[k, 6] = min(v / (v + vr), v / (v + vc))                
    #     pp[k, 7] = max(v / (v + vr), v / (v + vc))  
    #     pp[k, 8] = v / (v + vr)                
    #     pp[k, 9] = v / (v + vc)                
        
    #     aux_r = m[p[k, 0]].clone()
    #     aux_r[p[k, 1]] = torch.inf
        
    #     mask_r = stat_b[p[k, 1]]
    #     aux_r[mask_r] = torch.inf
        
    #     vr = aux_r.min()

    #     aux_c = m[:, p[k, 1]].clone()
    #     aux_c[p[k, 0]] = torch.inf
        
    #     mask_c = stat_a[p[k, 0]]
    #     aux_c[mask_c] = torch.inf
        
    #     vc = aux_c.min()

    #     if torch.isinf(vr): vr = v
    #     if torch.isinf(vc): vc = v

    #     pp[k, 10] = (2 * v) / (2 * v + vr + vc)                
    #     pp[k, 11] = min(v / (v + vr), v / (v + vc))                
    #     pp[k, 12] = max(v / (v + vr), v / (v + vc))  
    #     pp[k, 13] = v / (v + vr)                
    #     pp[k, 14] = v / (v + vc)                

    #     pp[k, 10] = pp[k, 10] / (1 - pp[k, 10])                
    #     pp[k, 11] = pp[k, 11] / (1 - pp[k, 11])                
    #     pp[k, 12] = pp[k, 12] / (1 - pp[k, 12])                
    #     pp[k, 13] = pp[k, 13] / (1 - pp[k, 13])                
    #     pp[k, 14] = pp[k, 14] / (1 - pp[k, 14])                

    idx = torch.argsort(pp[:, out_idx])
    m_idx = p[idx]
    val = pp[idx, out_idx]
            
    return m_idx, val


def plot_pair_matches(img, pt0, pt1, mask_dtm, mask_ransac):
    fig = plt.figure()    
    img0 = viz_utils.load_image(img[0])
    img1 = viz_utils.load_image(img[1])
    fig, axes = viz.plot_images([img0, img1], fig_num=fig.number)    
    viz.plot_matches(pt0[mask_dtm & mask_ransac], pt1[mask_dtm & mask_ransac], color='g', lw=0.2, ps=6, a=0.3, axes=axes, fig_num=fig.number)
    viz.plot_matches(pt0[mask_dtm & ~mask_ransac], pt1[mask_dtm & ~mask_ransac], color='r', lw=0.2, ps=6, a=0.3, axes=axes, fig_num=fig.number)
