# sources: inspired by https://plot.ly/python/filled-chord-diagram/

from __future__ import unicode_literals, print_function

import numpy as np

from plotly.graph_objs import *

from datadez.dataviz.color import polylinear_gradient

# Constants
PI = np.pi
GAP = 2 * PI * 0.005

COLOR_SCHEME = ["#64B5F6", "#E57373", "#AED581", "#FFB74D", "#90A4AE"]


def check_data(data_matrix):
    h, w = data_matrix.shape
    if h != w:
        raise ValueError('Data array must have (n,n) shape')
    return h


def modulo_a_b(x, a, b):
    # maps a real number onto the unit circle identified with
    # the interval [a,b), b-a=2*PI
    if a >= b:
        raise ValueError('Incorrect interval ends')
    y = (x - a) % (b - a)
    return y + b if y < 0 else y + a


def test_2_pi(x):
    return 0 <= x < 2 * PI


def get_ideogram_ends(ideogram_len, gap):
    ideo_ends = []
    left = 0
    for k in range(len(ideogram_len)):
        right = left + ideogram_len[k]
        ideo_ends.append([left, right])
        left = right + gap
    return ideo_ends


def make_ideogram_arc(radius, phi, a=50):
    # radius is the circle radius
    # phi is the list of ends angle coordinates of an arc
    # a is a parameter that controls the number of points to be evaluated on an arc
    if not test_2_pi(phi[0]) or not test_2_pi(phi[1]):
        phi = [modulo_a_b(t, 0, 2 * PI) for t in phi]
    length = (phi[1] - phi[0]) % 2 * PI
    nr = 5 if length <= PI / 4 else int(a * length / PI)

    if phi[0] < phi[1]:
        theta = np.linspace(phi[0], phi[1], nr)
    else:
        phi = [modulo_a_b(t, -PI, PI) for t in phi]
        theta = np.linspace(phi[0], phi[1], nr)
    return radius * np.exp(1j * theta)


def make_ribbon_ends(mapped_data, ideo_ends, idx_sort):
    length = mapped_data.shape[0]
    ribbon_boundary = np.zeros((length, length + 1))
    for k in range(length):
        start = ideo_ends[k][0]
        ribbon_boundary[k][0] = start
        for j in range(1, length + 1):
            J = idx_sort[k][j - 1]
            ribbon_boundary[k][j] = start + mapped_data[k][J]
            start = ribbon_boundary[k][j]
    return [[(ribbon_boundary[k][j], ribbon_boundary[k][j + 1]) for j in range(length)] for k in range(length)]


def control_pts(angle, radius):
    # angle is a  3-list containing angular coordinates of the control points b0, b1, b2
    # radius is the distance from b1 to the  origin O(0,0)

    if len(angle) != 3:
        raise ValueError('angle must have len =3')
    b_cplx = np.array([np.exp(1j * angle[k]) for k in range(3)])
    b_cplx[1] = radius * b_cplx[1]
    return zip(b_cplx.real, b_cplx.imag)


def ctrl_rib_chords(l, r, radius):
    # this function returns a 2-list containing control poligons of the two quadratic Bezier
    # curves that are opposite sides in a ribbon
    # l (r) the list of angular variables of the ribbon arc ends defining
    # the ribbon starting (ending) arc
    # radius is a common parameter for both control polygons
    if len(l) != 2 or len(r) != 2:
        raise ValueError('the arc ends must be elements in a list of len 2')
    return [control_pts([l[j], (l[j] + r[j]) / 2, r[j]], radius) for j in range(2)]


def make_q_bezier(b):
    # defines the Plotly SVG path for a quadratic Bezier curve defined by the list of its control points
    if len(b) != 3:
        raise ValueError('control poligon must have 3 points')
    A, B, C = b
    return 'M ' + str(A[0]) + ',' + str(A[1]) + ' ' + 'Q ' + \
           str(B[0]) + ', ' + str(B[1]) + ' ' + \
           str(C[0]) + ', ' + str(C[1])


def make_ribbon_arc(theta0, theta1):
    if test_2_pi(theta0) and test_2_pi(theta1):
        if theta0 < theta1:
            theta0 = modulo_a_b(theta0, -PI, PI)
            theta1 = modulo_a_b(theta1, -PI, PI)
            if theta0 * theta1 > 0:
                raise ValueError('incorrect angle coordinates for ribbon')

        nr = int(40 * (theta0 - theta1) / PI)
        if nr <= 2: nr = 3
        theta = np.linspace(theta0, theta1, nr)
        pts = np.exp(1j * theta)  # points on arc in polar complex form

        string_arc = ''
        for k in range(len(theta)):
            string_arc += 'L ' + str(pts.real[k]) + ', ' + str(pts.imag[k]) + ' '
        return string_arc
    else:
        raise ValueError('the angle coordinates for an arc side of a ribbon must be in [0, 2*pi]')


def make_layout(title, plot_size):
    axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )

    return Layout(title=title,
                  xaxis=XAxis(axis),
                  yaxis=YAxis(axis),
                  showlegend=False,
                  width=plot_size,
                  height=plot_size,
                  margin=Margin(t=25, b=25, l=25, r=25),
                  hovermode='closest',
                  shapes=[]  # to this list one appends below the dicts defining the ribbon,
                  # respectively the ideogram shapes
                  )


def make_ideo_shape(path, line_color, fill_color):
    # line_color is the color of the shape boundary
    # fill_collor is the color assigned to an ideogram
    return dict(
        line=Line(
            color=line_color,
            width=0.45
        ),

        path=path,
        type='path',
        fillcolor=fill_color,
        layer='below'
    )


def make_ribbon(l, r, line_color, fill_color, radius=0.2):
    # l=[l[0], l[1]], r=[r[0], r[1]]  represent the opposite arcs in the ribbon
    # line_color is the color of the shape boundary
    # fill_color is the fill color for the ribbon shape
    poligon = ctrl_rib_chords(l, r, radius)
    b, c = poligon

    b = list(b)
    c = list(c)

    return dict(
        line=Line(
            color=line_color, width=0.5
        ),
        path=make_q_bezier(b) + make_ribbon_arc(r[0], r[1]) +
             make_q_bezier(c[::-1]) + make_ribbon_arc(l[1], l[0]),
        type='path',
        fillcolor=fill_color,
        layer='below'
    )


def make_self_rel(l, line_color, fill_color, radius):
    # radius is the radius of Bezier control point b_1
    b = control_pts([l[0], (l[0] + l[1]) / 2, l[1]], radius)

    b = list(b)

    return dict(
        line=Line(
            color=line_color, width=0.5
        ),
        path=make_q_bezier(b) + make_ribbon_arc(l[1], l[0]),
        type='path',
        fillcolor=fill_color,
        layer='below'
    )


def map_data(data_matrix, row_value, ideogram_length):
    mapped = np.zeros(data_matrix.shape)
    for j in range(data_matrix.shape[0]):
        mapped[:, j] = ideogram_length * data_matrix[:, j] / row_value
    return mapped


def invPerm(perm):
    # function that returns the inverse of a permutation, perm
    inv = [0] * len(perm)
    for i, s in enumerate(perm):
        inv[s] = i
    return inv


def _plot_ideogram(layout, row_sum, labels, colors, ideo_ends):
    ideograms = []

    for k in range(len(ideo_ends)):
        z = make_ideogram_arc(1.1, ideo_ends[k])
        zi = make_ideogram_arc(1.0, ideo_ends[k])
        m = len(z)
        n = len(zi)
        ideograms.append(Scatter(x=z.real,
                                 y=z.imag,
                                 mode='lines',
                                 line=Line(color=colors[k], shape='spline', width=0.25),
                                 text='label: ' + labels[k] + '<br>' + 'occurrence: {:d}'.format(row_sum[k]),
                                 hoverinfo='text'
                                 )
                         )

        path = 'M '
        for s in range(m):
            path += str(z.real[s]) + ', ' + str(z.imag[s]) + ' L '

        Zi = np.array(zi.tolist()[::-1])

        for s in range(m):
            path += str(Zi.real[s]) + ', ' + str(Zi.imag[s]) + ' L '
        path += str(z.real[0]) + ' ,' + str(z.imag[0])

        layout['shapes'].append(make_ideo_shape(path, 'rgb(150,150,150)', colors[k]))

    return ideograms


def _plot_ribbon(layout, row_sum, labels, colors, ideogram_length, ideo_ends, matrix):
    ribbon_info = []

    radii_sribb = [0.9]  # these value are set after a few trials

    # Colors
    ribbon_color = [len(labels) * [colors[k]] for k in range(len(labels))]

    # Map ideogram and ribbons, sort by occurrence
    mapped_data = map_data(matrix, row_sum, ideogram_length)
    idx_sort = np.argsort(mapped_data, axis=1)

    ribbon_ends = make_ribbon_ends(mapped_data, ideo_ends, idx_sort)

    for k in range(len(labels)):
        sigma = idx_sort[k]
        sigma_inv = invPerm(sigma)
        for j in range(k, len(labels)):
            if matrix[k][j] == 0 and matrix[j][k] == 0:
                continue

            eta = idx_sort[j]
            eta_inv = invPerm(eta)
            l = ribbon_ends[k][sigma_inv[j]]

            if j == k:
                layout['shapes'].append(make_self_rel(l, 'rgb(175,175,175)',
                                                      colors[k],
                                                      radius=radii_sribb[k % len(radii_sribb)]))
                z = 0.9 * np.exp(1j * (l[0] + l[1]) / 2)
                # the text below will be displayed when hovering the mouse over the ribbon
                text = 'label %s appeared in %s samples' % (labels[k], matrix[k][k]),
                ribbon_info.append(Scatter(x=[z.real],
                                           y=[z.imag],
                                           mode='markers',
                                           marker=Marker(size=0.5, color=colors[k]),
                                           text=text,
                                           hoverinfo='text'
                                           )
                                   )
            else:
                r = ribbon_ends[j][eta_inv[k]]
                zi = 0.9 * np.exp(1j * (l[0] + l[1]) / 2)
                zf = 0.9 * np.exp(1j * (r[0] + r[1]) / 2)

                # texti and textf are the strings that will be displayed when hovering the mouse
                # over the two ribbon ends
                texti = '%s x %s: %s times' % (labels[k], labels[j], matrix[k][j])
                textf = '%s x %s: %s times' % (labels[j], labels[k], matrix[j][k])

                ribbon_info.append(Scatter(x=[zi.real],
                                           y=[zi.imag],
                                           mode='markers',
                                           marker=Marker(size=0.5, color=ribbon_color[k][j]),
                                           text=texti,
                                           hoverinfo='text'
                                           )
                                   ),
                ribbon_info.append(Scatter(x=[zf.real],
                                           y=[zf.imag],
                                           mode='markers',
                                           marker=Marker(size=0.5, color=ribbon_color[k][j]),
                                           text=textf,
                                           hoverinfo='text'
                                           )
                                   )

                # IMPORTANT!!!  Reverse these arc ends because otherwise you get
                r = (r[1], r[0])
                # a twisted ribbon
                # append the ribbon shape
                layout['shapes'].append(make_ribbon(l, r, 'rgb(175,175,175)', ribbon_color[k][j]))

    return ribbon_info


def chord_diagram(matrix, labels):
    label_count = check_data(matrix)

    # Useful vars
    layout = make_layout('Chord diagram', 600)
    row_sum = [np.sum(matrix[k, :]) for k in range(label_count)]

    # colors = polylinear_gradient(COLOR_SCHEME, len(ideogram_length))
    colors = polylinear_gradient(COLOR_SCHEME, label_count)
    ideo_colors = ['rgba(%s, %s, %s, 0.75)' % (r, g, b) for (r, g, b) in zip(colors['r'], colors['g'], colors['b'])]

    ideogram_length = 2 * PI * np.asarray(row_sum) / sum(row_sum) - GAP * np.ones(len(labels))
    ideo_ends = get_ideogram_ends(ideogram_length, GAP)

    ideograms = _plot_ideogram(layout, row_sum, labels, ideo_colors, ideo_ends)
    ribbon_info = _plot_ribbon(layout, row_sum, labels, ideo_colors, ideogram_length, ideo_ends, matrix)

    data = Data(ribbon_info + ideograms)
    figure = Figure(data=data, layout=layout)

    return figure
