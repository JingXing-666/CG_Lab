import taichi as ti
import taichi.math as tm
from config import v0,v1,v2;

ti.init(arch=ti.cpu)

vertices = ti.Vector.field(3, dtype=ti.f32, shape=3)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=3)

@ti.func
def get_model_matrix(angle: float):
    '''
    该函数用于接收一个旋转角度（角度制），返回绕 Z 轴旋转该角度的模型变换矩阵
    '''
    rad = angle / 180.0 * tm.pi
    cos_rad = tm.cos(rad)
    sin_rad = tm.sin(rad)
    model_matrix = ti.Matrix([
        [cos_rad, -sin_rad, 0.0, 0.0],
        [sin_rad, cos_rad, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return model_matrix

@ti.func
def get_view_matrix(eye_pos: ti.template()):
    '''
    接收相机位置（三维向量），返回视图变换矩阵
    '''
    view_matrix = ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return view_matrix

@ti.func
def get_projection_matrix(eye_fov: float, aspect_ratio: float, zNear: float, zFar: float):
    '''
    接收视场角(Y 轴方向，角度制）、屏幕长宽比、近截面距离和远截面距离，返回透视投影矩阵
    '''
    n=-zNear
    f=-zFar
    #视锥体转正交长方形
    persp_to_ortho_matrix = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n+f, -n*f],
        [0.0, 0.0, 1.0, 0.0]
    ])

    t = tm.tan(eye_fov/180.0*tm.pi / 2.0) * zNear
    b = -t
    r = aspect_ratio * t
    l = -r
    #正交投影矩阵
    ortho_maitrix = ti.Matrix([
        [2.0/(r-l), 0.0, 0.0, -(r+l)/(r-l)],
        [0.0, 2.0/(t-b), 0.0, -(t+b)/(t-b)],
        [0.0, 0.0, 2.0/(n-f), -(n+f)/(n-f)],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return ortho_maitrix @ persp_to_ortho_matrix

@ti.kernel
def compute_transform(angle: float):
    '''
    计算mvp变换后的屏幕坐标
    '''

    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model_matrix = get_model_matrix(angle)
    view_matrix = get_view_matrix(eye_pos)
    projection_matrix = get_projection_matrix(45.0,1.0,0.1,50.0)
    mvp = projection_matrix @ view_matrix @ model_matrix

    for i in range(3):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        #模型变换
        v_clip = mvp @ v4

        #透视除法
        v_homogeneous = v_clip / v_clip[3]

        #视口变换
        screen_coords[i] = ti.Vector([
            (v_homogeneous[0] + 1.0) / 2.0,
            (v_homogeneous[1] + 1.0) / 2.0
        ])

def main():
    #初始化三角形顶点:
    vertices[0] = v0
    vertices[1] = v1
    vertices[2] = v2

    #创建GUI窗口
    gui = ti.GUI("3D Transformations", (800, 800))
    angle = 0.0

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle += 10.0
            elif gui.event.key == 'd':
                angle -= 10.0
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        #计算变换
        compute_transform(angle)

        #获取2D屏幕坐标并绘制
        a = screen_coords[0]
        b = screen_coords[1]
        c = screen_coords[2]

        gui.line(a,b,radius=2,color=0xFF0000)
        gui.line(b,c,radius=2,color=0xFF0000)
        gui.line(c,a,radius=2,color=0xFF0000)

        gui.show()

if __name__ == "__main__":
    main()