from PFRecAlg3d import *
from utils_3d_test_function import *


dtlz1 = DTLZ1(
    x_range=((0, 1), (0, 1), (0, 1)),
    k=1
)

dtlz2 = DTLZ2(
    x_range=((0, 1), (0, 1), (0, 1)),
    k=1
)

eps = 0.0
dtlz3 = DTLZ3(
    x_range=((0., 1.), (0., 1.), (0.+eps, 1.-eps)),
    k=1
)

eps = 0.95
dtlz4 = DTLZ4(
    x_range=((0+eps, 1), (0+eps, 1), (0, 1)),
    k=1,
    alpha=100
)

dtlz5 = DTLZ5(
    x_range=((0, 1), (0, 1), (0, 1)),
    k=1,
    alpha=0.1
)

dtlz6 = DTLZ6(
    x_range=((0, 1), (0, 1), (0, 1)),
    k=1
)


def generate_test(
        prob,
        sample_num=20,
        resolution=(256, 256, 256),
        y_clip_range=((0, 2), (0, 2), (0, 2))
) -> TestPro:

    test = TestPro(
        prob,
        sample_num=sample_num,
        resolution=resolution,
        y_clip_range=y_clip_range
    )

    test.run()
    test.show3d(test.y_samples, 1)
    test.show3d(test.y_index_list, 1)

    return test


def recognize(t: TestPro, u=3):

    f_rec = PFRecAlg3d(u)
    f_rec.get_input(t.y_matrix)
    f_rec.run()
    f_rec.show()

    return f_rec


t = generate_test(
    prob=dtlz1,
    sample_num=50,
    resolution=(256, 256, 256),
    y_clip_range=((0, 2), (0, 2), (0, 2))
)
f_rec = recognize(t, u=3)
