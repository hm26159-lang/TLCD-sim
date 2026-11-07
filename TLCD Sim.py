"""TLCD 형상·설치 방향 실험. 실행: python TLCD_3D.py

NumPy, Matplotlib만 사용한다. 모든 계산 단위는 m, kg, s이다.
건물은 수평 두 축으로 움직이고 물은 U자 관을 따라 이동한다.
지붕은 회전하지 않으며 건물의 두 수평 방향 강성과 감쇠는 같다.
"""
from dataclasses import dataclass, replace, asdict
from pathlib import Path
import argparse
import csv
import json
import math
import numpy as np

G = 9.80665


@dataclass(frozen=True)
class Building:
    mass: float = 20.0       # 물과 관을 제외한 모형 구조물 질량 [kg]
    frequency: float = 1.0   # 장치가 없는 구조물의 고유진동수 [Hz]
    damping: float = 0.02    # 구조물 감쇠비: 0.02 = 2%
    container_mass: float = 0.0  # 관 자체 질량 [kg], 실제 측정값으로 변경

    @property
    def stiffness(self):
        return self.mass * (2 * np.pi * self.frequency) ** 2

    @property
    def viscous(self):
        return 2 * self.damping * self.mass * 2 * np.pi * self.frequency


@dataclass(frozen=True)
class Damper:
    B: float = 0.30          # 바닥 수평 관의 중심선 길이 [m]
    h: float = 0.10          # 바닥 중심선부터 정지 수면까지 높이 [m]
    diameter: float = 0.05   # 원형 관의 안지름 [m], 모든 구간 일정
    angle: float = 0.0       # 지진 방향과 관의 수평축 사이 각도 [도]
    loss: float = 5.0        # 전체 손실계수. 형상별 실험 보정이 필요
    freeboard: float = 0.12  # 정지 수면 위 여유 높이 [m]

    shape: str = 'U형'
    tilt: float = 60.0  # V형 다리와 수평면 사이 각도
    reverse: bool = False

    @property
    def area(self):
        return np.pi * self.diameter ** 2 / 4

    @property
    def length(self):
        return self.B + 2 * self.h

    @property
    def mass(self):
        return 1000 * self.area * self.length  # 물의 밀도 1000 kg/m³

    @property
    def coupling(self):
        # 설치 방향은 관 자체 길이/진동수를 바꾸지 않고 힘의 결합을 바꾼다.
        return 1000 * self.area * self.B * math.cos(math.radians(self.angle))

    @property
    def frequency(self):
        if self.shape!='U형':
            mf,kf=network(self)[7:9]
            return math.sqrt(np.linalg.eigvals(np.linalg.solve(mf,kf)).real.min())/(2*np.pi)
        return math.sqrt(2 * G / self.length) / (2 * np.pi)

    @property
    def limit(self):
        # 수면이 바닥에 닿거나 관 밖으로 넘치면 현재 모델을 적용할 수 없다.
        return min(self.h, self.freeboard)


def validate(b, d):
    if not all(np.isfinite(list(asdict(b).values()))) or not all(np.isfinite([v for k,v in asdict(d).items() if k!="shape"])):
        raise ValueError('모든 매개변수는 유한한 수여야 합니다.')
    if d.shape not in SHAPES or not 0<d.tilt<=90: raise ValueError('관 종류와 V형 각도(0도 초과~90도)를 확인하세요.')
    if min(b.mass, b.frequency, d.B, d.h, d.diameter, d.freeboard) <= 0:
        raise ValueError('질량, 진동수, 길이와 지름은 양수여야 합니다.')
    if min(b.damping, b.container_mass, d.loss) < 0 or not 0 <= d.angle <= 180:
        raise ValueError('감쇠와 관 질량은 0 이상, 각도는 0~180도여야 합니다.')


def demo_input(duration=20.0, dt=0.005, pga=0.03, seed=7):
    """재현 가능한 합성 진동. 실제 지진 기록/설계 지진이 아니다. pga 단위는 g."""
    t = np.linspace(0, duration, round(duration / dt) + 1)
    rng = np.random.default_rng(seed)
    frequencies = np.linspace(0.4, 4.0, 45)
    weights = np.exp(-((frequencies - 1.2) / 0.9) ** 2)
    raw = np.sum(weights[:, None] * np.sin(2 * np.pi * frequencies[:, None] * t
                 + rng.uniform(0, 2*np.pi, (45, 1))), axis=0)
    raw *= np.sin(np.pi * t / duration) ** 2
    return t, raw / np.max(np.abs(raw)) * pga * G


def recorded_input():
    """1940 El Centro N-S, Peknold판. 원본의 0.02초 간격과 소수점 5자리 g 보존.
    https://github.com/OpenSees/OpenSees/blob/master/EXAMPLES/ExampleScripts/elCentro.AT2
    기본 재생은 파형을 보존하고 최대 가속도만 0.03 g로 맞춘다.
    """
    import base64, zlib
    packed = (
        'c-k%4dAyHh`Umh^C$eNogsj<$(#e*}c0$X9${-ny?1O1&LoZ8NDr)>F5=SGUL}gMaN+C;QnaGl*Le>!HWOs1x-}{+we_XF;x'
        'v%AOeXi?Xo@d|{Av}>Mgzh1Pk6h1sZhB=1AGntEJUqtuq-&S75FXAO!glXF(|y;wpIu9mrHQ@qg2q3|7sA}!7~eS?W2H;3#)'
        '`2S;QfyUL&zu^!s`V>*iU~2d7BI+*RxrCKE_8*#rWgN7~eY*;{zvRJmLNY?~0v{aR#Y#CdS{gVw`<6#x4hAynb(t&0PC<e$D'
        '&8k{aYi@%LPg@h+GqT^&O9wIQ@D8^Y|$^1LR51#mB+w`_NeYyKPKhNUsS;o8abT|dWI`llGr?(*(OG2Q35U%4~JsdUQii*Y*'
        'tHxvqCR>csi*9&3J{UOx+YY0c$hw%P`A@sa2gsK@K?7Jg`qD@0+d`}3|>xPg={2s+ZSR>z?j>q^a|Hs$H_|W1QKbak4iAgaI'
        '92H~6+c6eg662OFF;2t7-*8sDY6usbgs{3}2;-j(VcK8RNsPtS`IF;TT<zNtWA@iEUVb;mLjRPHI3GTx#vMYqwQ&ep;?K_?!'
        'c%<Kyh<J5e+$NTZ9*vjKnMq$g|MuC2$?s9@C?3M7YX4;HOzzijrjZ*o4)vKQ6_{6H-vDG-c_YSD23-6)#>(M@B>GhnjXRVnY'
        '5Jc)dz68HO8u|V*E}{&+U(~GX3>>qt4+NGuOu$=EvA%V~j&!TT)mr-V#E+Rv{F=ErdgSEjt(E6ntgsfw%P0v>G^V9>TTsMwb'
        'ZTl772M@3vEy5yeB82jf9_E2vcm@z1FD#Ebe;KJ)ZrL!4dDXAaCeH|W35>3pl+IPCW;-t^<IxGKl*!)&L~J!U@K;a%&T7`y!'
        'z;~@Pqc9Y)xCdSVe%0~?=s@pU2`Bop5&5p6Z99O|t`bKrQQ%v>x4}1gpD(U?~9JW=PQ))1wm;O+L+be`nK#U)^#W?Si7@ybU'
        '56NdV+{^HGPT!}0r!T(Li*sV!u5U`h&=z+St`A|;z2>gE-c~b&U)1q+e120cgwx_wdnAPC^yfO`<&5_?z5*8)_^%CPAN|}Fr'
        'jF%8XrovEgwwl|xvB0?&>w;G`NsF~Ss0c^_?oGn`Ek?_=B;|66#b6o<iE_F-EiKpBF63u^!1t;%bQ0R5cz}N>_YF@?=cSW`~'
        'myj{LR%b1Jrnno<Azz_we1t+A&Mr4${97r+xLzcy(PU)=nG_rCW)=m3nWBKHbOWeZ2oe?dsCI!1veIl-=xe^~4&!#_5F`YOs'
        'o4w&(K16bMJz|I=6>&-c--CeJkew^v_%g^Q`i2l4s1*eyLb(BoD3xW%=;=T^qM8y_s@>-uPp{`%Vbu?*(ra$C;t3iIl?+{%c'
        '5kG_iT7hlJ>_e0%xmP-a7mDTfz@%d~9iGNniA~02#Um5WV(Yq|CEWB-W{aEbru+(6?QoeO@HPd}@??=!d&d&?NJ@0v_99QGH'
        '2%QdUbyBRc{A@5+a@p2~zX!Z;UWjq6IG5Ppis!B{PQ=HH?g#MIi;wljZ-Z%)9?8*v|0kD9dVB%y%9GLh>v#A+b$!-zeLh;q<'
        '<IKVRm_&+9B~~(4)B*DmuY(Gg7FTni*Rs?ejT-b(i--$HQ~6m@XI<Oj4_|ui&>e^tURf`sUs;&cH@7(`8mS+dd9V{y!LqCj{'
        'aJ<iz<aMgI#lT;Go`bu776O3mUJBao2Whv3XY<rWJbhNjzP|+spE}#$0);dTQ;jYW=$b#){^9`Q>;JvxRw6G}-fTJe%!PYTw'
        'xSIy`rf+X1;eg~MF&r@Lm*{l)u{=0O!}%6M^etS`Or@;x7o(^7lJn_~AW9l~{ZUqh!BOmB#tqwlKG*<^eOj?$Ao7S<{9X(e_'
        'Z^Kg;;bNRm4zB8WB9dvHsYpJ#LQ9adL?ViKK80*B8^bo!?mRBz|glPi~I;%yNI3v_=i#k@jI@OcY*?o(Htn%i)J*uDDuTIuY'
        'G3KzZYwj+iQ$>z1!+KPowYFaLg?%)P?;2muzP&me<^KhCCE)I4Y$iWP@l)GAl?K~3zTekZgXy+n_qFHRRn#2b_Ud<4#ne3NW'
        '4x?*8~Au$Kc&lgufCpuhvxR7xH-lzzE@wI)UoG`T^8fvkFBNF&<@tzqH6dptS`v5ff`LIpX%$!a8gnK?Z4Ss!ag}(+~@2=U1'
        '4~DPD}Ay*%w=T&Y)X|eKj1E(!Wd9zL}iY!t+p)qk8XD-)ZX8NWMSf_ail5tzY(-vpc+-OJ|6>x57(#`Oeli>H2pu-Hon?<ak'
        '*PO5k?@u6~eOgJhq?@j8Ak^PkQ~H}Rg~^Cj;li;>Sf7@&U+TASMGiIvu;p>W(zKg$|82kvw}7wM(m>ReX*jeLyM2le&C7j!c'
        '4yh7hJvp#$yw^qit!_w8ham=24kLz^L=kd5SsjHfN#qUM_8nb%={zu3Vc;8W<)q3Kny`iX@jN0S;C5Jw8|C!ChxLw6(NBiOl'
        '{W^+WIr14CgTx-HC;kR&TbP>Qy{1}~6#sRc6^8E>x#dO9aOaQj?MIcxc}Bcd;+DiuFE#ocu1@Aqr2nou-OnbUd{@ITil2|<u'
        '+{xT;;y52PqG)|e2}@GF6Lpip9E_QJnWbA2+|8z#m=Vq>Vb<Q@GaI)yVzb-heLATd_{_vY`t1l&K=ZalV1H3+i%pXA$$wPt)'
        '=#}`DhKtDfOSKkN(Rh@KqPLz4-1Rhx_@vgG~dyP~TXfyCJE&eA>zP8aXWx`;kL2p2zc8_mwWB_R{<D_!ExD6;G{6C%vD_Pjh'
        '3(aPWp)8zu8wya(0L68XP9yghsmz}1vaJv!x$<xAEpv7Z*ZwOBjg?ZHo$It@<FviuCQ2Ib;wxV%@pF5qhhz43bgPP`ouf1Dh'
        'P^79%#H|I_D`e=OB!Try29b>#MyXby{_ie~_JUm9a^WRq<Kd4u;<lG0-RCa%--_RVt+S#S!FRAZ|(|Ea?uW|IJ!atkd0`}Lq'
        'zn%L|YIT(TbH?+@f0B=%D*Cda+PAYtJ>gw{dgtKF(xZpq_*uPb7D@HUN}LvB)0Xa=bav1wYpgSzN&oeSX%HTUtNobdya3l37'
        ';n*w-Neomd!zi1C;NV~$1Cl9-u1*yC)^dr&vHJVrdz|iy~ZcgPx>|g<<wY3*JR%9_P&W)o#*RQ@19^Y8ul4-{SvN%aCemF9J'
        ';Ic0VU>&m3*f)px>7NC&e5qW=%Ev5Bpx~m8Nzx@Yq!SCOSVH)>~!Vw=_PC&By%h<98kWpYoN-zA`KyidoN`-HM}U#lBWAEui'
        '0)zf*d49-qC%sSL+Y#)iON2B)9na}oYNmB)O(dXq|I2i@`Fk77Rv&cBQMxZK*R$<6fB#OO)BB{e+1XgpbS?}E34em#CBwf@f'
        '#Z*cNWp$}@QRVnv*yg$bGHtXkHb95N}Kg+K=UUKC1oxYhxcZQtaf$=T1o-D5k>b}B#w)c$+rOxtO#I3+?tT;LDd-MN;_q9Ba'
        ';&W$K>OAqX^Zq=2vCaJ@7^{ex2Jb#tRwQ*}Hyp+;;x-lMZT>Ig<S`sBwWgQF;b3^LNWM8e7f;rY<lIW{L)fOUeMy~@-g#UPw'
        'xT;uoK1Kyj=y&DnnY*2e9GwePS(0<&Oh~>gVyuWO)bys)A9RLXWoh8Ogfo5+ub7GdGaisY&E!3o+EQoXNr>aI$Ik~<MCBFRT'
        '6&@Y%Rt6h3>2BTsN6VIC~k+boJioxj&8`Ax%hia*V#u$2fqkPx1flS0~u7fNc`nH`z>fpDBkl_3nqKopNc8tAF!ZT23d`VFU'
        'ab*2zWIgPM4l4%1BX9+^YRCVj<!ni>z}_oy}OK!Pu+8=f}Fd9A&*FWW-$n<t+KFQxX>+u(azj0N;g^YgIlJYxm$(#zONI(3t'
        'OEAIRA*RVipo!Kn!Y0gJ|>}M6_u#N3l9M@H!UHW50ay})=clO<6iFsaIoVj?eoUCOyEM7FV7vBPR1%8VaOwEBa<kw{1GY6CJ'
        '`(}EoOEOpam@3Za>U1rf{YVG6?$$HOe}ik;U4rcpZ11`DD406m?GtkbKV8_KmfINll#|PRHZ9oXm+N|cvP+yJ#?Pr+K6qBTj'
        '_};X`%7|Kg2O3t86;-=WKU6}w1f*W-bwg@uZemj=c;UWYuP7f#UXUs*<Wf|OY+%g3(GCp-^-ahRh%mt*<0KHmikT|P5+%_Ke'
        'iWrUnjK|_Y}8@o-D<8w)pGV&PwKx=dJqmu-?eQeG&a}pX+e%exv`W^<pKx^yK?cp3THPN+#%yyZGI0-j9^Sjmdmb$NsRD^nN'
        'K`yNNl6j}v0lOTHUnE2WPXdf&#v+eOU&#)ql#*gy8;WbP(=R1zywZ8P<8rXI<3M$R-hGw2+a>lk_0gmnW>A2!Ff!TT5guXg('
        'u>Fx~lw0S;K+}G7?hB|x#<8tTMO|`6t{#6um)@^|Q?(n>3Pin#6d00NedsnqCDgIh~->k0V;LOHxX?`={?hXHId@qK1Z?XsR'
        'bDMs7RBeXyJIB9-O?W@$dY$v~ov?S3$87mz^V?8;UdHFua-G256b)~SQx(<^E~fgciRUrNaX#rYJ+Q*Np0E{wcb+=8f$0I)='
        '5RE^ZGCpPhE$(U=C`H3PrkhuiuH`=I{f{v=JRn~F4+^+B3TzY(_afqeR|W_<fYp$q`v+0uwS1)_3v{sondfvq0`K}s`9u>os'
        'U>+M(TyKI9Nch6}yAUcR0KwU0-7J3Y&9k+OuHl`|55ye5`L8u>ab-dyRd^&hMMB9<B^Mw@6*DbZzT-CO_GKoWZ7wahlJEyqh'
        'cME9LZfvNo7|1MyH-+~l9fyySl#T=`&~Nxz9)cgcI;$<%+grR?^wzapmgnc-qLWtU41C*PaNzaCe=KEo`w$$37%cVF`N4>?q'
        'l_e6NE@@_Eu6MCU59NVmq2h`;B=@hmSVpPR_UG~l8u!GMh(vpSN`VH4+?~20ON{_yR(={;f<u}K>5`52vcdYmmNhX<2W|NOd'
        'a=uwYmXX!T8c#Np9hX!8h!2wAE~n~zo<x!tmgL_`aZ;L8Ak|3H=e5ZlxvAc3L0XbFqytI*+w>rP$x~zy8Ae_tuaeiuWHOD+A'
        '|H_X<TLVB^8YF~^{-`9ZtDNoPh>wiLbAx&<UdSqsvZSM(c~W@DHl^`qnk*2a_%P$Ni%Xkc`#W=lQS&oMtYNe$$m+OkP$I8kH'
        '(QV$XhXW_IaQDoA|XEzxbmE{34BhgU94E+L93WCTk6`EW}gEIfkfQ_<wX)j-&'
    )
    raw=np.frombuffer(zlib.decompress(base64.b85decode(packed)),dtype='<i4').astype(float)*1e-5
    return np.arange(len(raw))*.02, raw*(.03*G/np.max(np.abs(raw)))


def load_input(path, unit):
    """헤더 time_s,accel의 CSV를 읽는다. 단위는 명시적으로 선택한다."""
    data = np.genfromtxt(path, delimiter=',', names=True, encoding='utf-8-sig')
    if data.dtype.names != ('time_s', 'accel'):
        raise ValueError('CSV 첫 줄은 time_s,accel 이어야 합니다.')
    t, a = np.atleast_1d(data['time_s']), np.atleast_1d(data['accel'])
    if len(t) < 3 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(a)):
        raise ValueError('CSV에 유효한 숫자 행이 3개 이상 필요합니다.')
    if np.any(np.diff(t) <= 0):
        raise ValueError('시간은 중복 없이 증가해야 합니다.')
    factor = {'m/s2': 1.0, 'g': G, 'cm/s2': 0.01}[unit]
    return t - t[0], a * factor


def metrics(result):
    # 넘침/바닥 도달 후에는 전체 기록의 저감률을 보고하지 않는다.
    if not result['valid']:
        return None
    x = result['state'][:, [4, 6, 0]]
    peak = np.max(np.abs(x), axis=0)
    rms = np.sqrt(np.trapezoid(x*x, result['t'], axis=0) / np.ptp(result['t']))
    acc = np.max(np.abs(result['acc']), axis=0)
    def reduction(values, base):
        return float(100 * (1-values[2]/values[base])) if values[base] > 1e-12 else None
    return {'peak_displacement_m': peak.tolist(), 'rms_displacement_m': rms.tolist(),
            'peak_absolute_acceleration_m_s2': acc.tolist(),
            'peak_reduction_vs_bare_pct': reduction(peak, 0),
            'peak_reduction_vs_fixed_pct': reduction(peak, 1),
            'rms_reduction_vs_bare_pct': reduction(rms, 0),
            'acceleration_reduction_vs_bare_pct': reduction(acc, 0),
            'peak_water_displacement_m': float(np.max(np.abs(result['state'][:, 2])))}


def setup_plot():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ['Malgun Gothic', 'AppleGothic', 'NanumGothic']:
        if name in names:
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def draw_geometry(ax, d):
    ax.clear()
    top = d.h + d.freeboard
    ax.plot([-d.B/2, -d.B/2, d.B/2, d.B/2], [top, 0, 0, top],
            lw=12, color='#cbd5e1', solid_capstyle='butt')
    ax.plot([-d.B/2, -d.B/2, d.B/2, d.B/2], [d.h, 0, 0, d.h],
            lw=6, color='#0284c7', solid_capstyle='butt')
    ax.axhline(d.h, color='#0284c7', ls=':', lw=1)
    ax.text(0, top, f'B={d.B:.2f} m | h={d.h:.2f} m\n안지름={1000*d.diameter:.0f} mm', ha='center', va='bottom', fontsize=9)
    ax.set(xlim=(-d.B/2-0.12, d.B/2+0.12), ylim=(-0.06, top+0.13),
           title='U자 관 측면 · 파란색 = 정지 상태의 물', xlabel='수평 거리 [m]', ylabel='높이 [m]')
    ax.set_aspect('equal', adjustable='box')


def draw_result(fig, axes, b, d, r, title):
    geometry, plan, disp, accel, water, info = axes
    for ax in axes:
        ax.clear()
    if d.shape=='U형':draw_geometry(geometry,d)
    else:
        feet,up,leg,starts,ends=network(d)[:5]
        geometry.remove();geometry=fig.add_subplot(3,2,1,projection='3d')
        for start,end in zip(starts,ends):
            geometry.plot(*np.stack((start,end)).T,color='#258aae',lw=5)
        for i,(start,direction) in enumerate(zip(feet,up)):
            top=start+direction*(leg+d.freeboard/direction[2])
            geometry.plot(*np.stack((start,top)).T,color='#9bb2bf',lw=5)
            geometry.plot(*np.stack((start,start+direction*leg)).T,color='#258aae',lw=3)
            geometry.text(*top,str(i+1))
        geometry.set_title(d.shape+' · 기둥 번호');geometry.set_axis_off()
    theta = math.radians(d.angle)
    plan.annotate('', xy=(1, 0), xytext=(-1, 0), arrowprops=dict(arrowstyle='->', color='#ef4444', lw=2))
    plan.plot([-math.cos(theta), math.cos(theta)], [-math.sin(theta), math.sin(theta)], color='#0284c7', lw=5)
    plan.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), title=f'위에서 본 설치 방향: θ={d.angle:.0f}°')
    plan.text(0, -1.18, '빨강: 지진 방향 / 파랑: 관 수평축', ha='center', fontsize=9)
    plan.set_aspect('equal'); plan.axis('off')
    labels = ['장치 없음', '동일 질량 고정', 'TLCD']
    colors = ['#64748b', '#f59e0b', '#0284c7']
    for j, column in enumerate([4, 6, 0]):
        disp.plot(r['t'], 1000*r['state'][:, column], label=labels[j], color=colors[j], lw=1.1)
        accel.plot(r['t'], r['acc'][:, j]/G, color=colors[j], lw=1.0)
    disp.set(title='건물의 지면 대비 변위', ylabel='변위 [mm]'); disp.legend(fontsize=8, ncol=3)
    accel.set(title='건물 절대가속도', ylabel='가속도 [g]')
    if 'levels' in r:
        for j,level in enumerate(r['levels'].T):water.plot(r['t'],1000*level,label=f'기둥 {j+1}')
        water.legend(fontsize=8);water.set(title='각 기둥의 수면 높이 변화',ylabel='수면 변화 [mm]')
    else:
        water.plot(r['t'], 1000*r['state'][:, 2], color='#0284c7')
        for sign in [-1, 1]:
            water.axhline(sign*1000*d.limit, color='#dc2626', ls='--')
        water.set(title='오른쪽 수면 변화 · 왼쪽은 반대', ylabel='수면 변화 [mm]')
    for ax in [disp, accel, water]:
        ax.set_xlabel('시간 [s]'); ax.grid(alpha=0.2)
        if not r['valid']:
            ax.axvline(r['failure_time'], color='red', ls=':')
    info.axis('off')
    m = metrics(r)
    text = (f'물 질량: {d.mass:.3f} kg ({100*d.mass/b.mass:.1f}%)\n'
            f'관 질량: {b.container_mass:.3f} kg\n'
            f'건물 / 액체 고유진동수: {b.frequency:.3f} / {d.frequency:.3f} Hz\n'
            f'전체 물기둥 길이 L: {d.length:.3f} m\n')
    if m is None:
        text += f"\n{r['failure_time']:.3f}초: 수면 한계 도달\n계산 중단 · 전체 저감률 산출 불가"
    else:
        fmt = lambda v: '정의 불가' if v is None else f'{v:+.1f}%'
        text += '\n최대 변위 [mm] (없음 / 고정 / TLCD)\n' + ' / '.join(f'{v*1000:.2f}' for v in m['peak_displacement_m'])
        text += f"\n\n최대 변위 저감 (장치 없음 대비): {fmt(m['peak_reduction_vs_bare_pct'])}"
        text += f"\n최대 변위 저감 (고정 질량 대비): {fmt(m['peak_reduction_vs_fixed_pct'])}"
        text += f"\nRMS 변위 저감 (장치 없음 대비): {fmt(m['rms_reduction_vs_bare_pct'])}"
        text += f"\n최대 가속도 저감 (장치 없음 대비): {fmt(m['acceleration_reduction_vs_bare_pct'])}"
        text += '\n\n양수 = 감소 / 음수 = 악화'
    info.text(0, 1, text, va='top', fontsize=9)
    fig.suptitle('TLCD 형상·방향 실험 | '+title, fontsize=13)


def save_result(folder, b, d, r, fig, title):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder/'response.png', dpi=160)
    np.savetxt(folder/'time_history.csv', np.column_stack([r['t'], r['ag'], r['state'], r['acc']]),
               delimiter=',', header='time_s,ground_accel_m_s2,x_tlcd_m,v_tlcd_m_s,water_y_m,water_v_m_s,x_bare_m,v_bare_m_s,x_fixed_m,v_fixed_m_s,acc_bare_m_s2,acc_fixed_m_s2,acc_tlcd_m_s2', comments='')
    payload = {'building': asdict(b), 'damper': asdict(d), 'input': title,
               'valid': r['valid'], 'failure_time_s': r['failure_time'],
               'metric_array_order': ['bare', 'fixed_mass', 'tlcd'], 'metrics': metrics(r)}
    payload['installed']=r.get('installed',True)
    if not payload['installed']: payload['damper']=None
    payload['source'] = r.get('source')
    (folder/'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def sweep(b, d, t, ag, folder):
    """총 물 길이와 지름을 유지하면서 B/L과 설치 각도를 비교한다."""
    plt = setup_plot()
    ratios = np.array([0.35, 0.50, 0.65, 0.80])
    angles = np.arange(0, 91, 15)
    values = np.full((len(ratios), len(angles)), np.nan)
    rows = []
    for i, ratio in enumerate(ratios):
        for j, angle in enumerate(angles):
            test = replace(d, B=d.length*ratio, h=d.length*(1-ratio)/2, angle=float(angle))
            r = simulate_space(b, test, t, ag); m = metrics(r)
            if m is not None and m['peak_reduction_vs_bare_pct'] is not None:
                values[i, j] = m['peak_reduction_vs_bare_pct']
            rows.append({'B_over_L': ratio, 'angle_deg': angle, 'B_m': test.B, 'h_m': test.h,
                         'water_mass_kg': test.mass, 'valid': r['valid'],
                         'peak_reduction_vs_bare_pct': values[i, j],
                         'peak_reduction_vs_fixed_pct': m['peak_reduction_vs_fixed_pct'] if m else '',
                         'acceleration_reduction_vs_bare_pct': m['acceleration_reduction_vs_bare_pct'] if m else ''})
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    with (folder/'sweep.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
    finite = np.abs(values[np.isfinite(values)])
    bound = max(float(np.max(finite)) if len(finite) else 1, 1)
    cmap = plt.get_cmap('RdBu').copy(); cmap.set_bad('#dddddd')
    im = ax.imshow(values, cmap=cmap, vmin=-bound, vmax=bound, aspect='auto')
    for i in range(len(ratios)):
        for j in range(len(angles)):
            ax.text(j, i, f'{values[i,j]:+.1f}%' if np.isfinite(values[i,j]) else '한계 초과', ha='center', va='center', fontsize=9)
    ax.set(xticks=range(len(angles)), xticklabels=angles, yticks=range(len(ratios)), yticklabels=ratios,
           xlabel='지진 방향과 설치 방향 사이 각도 [도]', ylabel='가로 길이 비율 B/L',
           title='최대 건물 변위 저감률 · 장치 없음 대비\n총 물 길이·물 질량 일정 / 양수: 감소, 음수: 악화')
    fig.colorbar(im, ax=ax, label='저감률 [%]')
    fig.savefig(folder/'sweep.png', dpi=160)
    plt.close(fig)




SHAPES=('U형','피자 거치대','V형','다기둥형')

def network(d):
    """각 가지의 물속 길이와 방향. 분기점에서 들어온 부피와 나간 부피가 같다."""
    n={'피자 거치대':3,'다기둥형':4}.get(d.shape,2)
    leg=2*d.h/n
    surface=np.vstack([np.eye(n-1),-np.ones(n-1)])
    if d.shape=='피자 거치대':
        az=np.arange(n)*2*np.pi/n
        feet=d.B/n*np.column_stack((np.cos(az),np.sin(az),np.zeros(n)))
        starts=np.zeros_like(feet); ends=feet.copy(); paths=surface
    else:
        feet=np.column_stack((np.linspace(-d.B/2,d.B/2,n),np.zeros((n,2))))
        starts=feet[:-1];ends=feet[1:];paths=-np.cumsum(surface,axis=0)[:-1]
    up=np.tile([0.,0.,1.],(n,1))
    if d.shape=='V형':
        tilt=np.radians(d.tilt)
        up[:,0]=[-np.cos(tilt),np.cos(tilt)];up[:,2]=np.sin(tilt)
    delta=ends-starts
    length=np.linalg.norm(delta,axis=1)
    rhoA=1000*d.area
    inertia=rhoA*(leg*surface.T@surface+paths.T@(length[:,None]*paths))
    spring=rhoA*G*surface.T@(up[:,2,None]*surface)
    coupling=rhoA*(delta[:,:2].T@paths+leg*up[:,:2].T@surface)
    return feet,up,leg,starts,ends,surface,paths,inertia,spring,coupling


def network_response(b,d,t,ag):
    """수평 두 축의 2층 건물과 연결된 물기둥들의 결합 운동을 적분한다."""
    feet,up,leg,starts,ends,S,T,fluid_mass,fluid_k,coupling=network(d)
    modes=S.shape[1]
    size=8+modes
    # 0~3: X/Y 각 층, 4~7: 맨 건물/고정 질량, 8 이후: 독립 유체 좌표
    floor=b.mass/2; added=d.mass+b.container_mass
    M=np.eye(size);K=np.zeros_like(M);C=K.copy()
    omega=2*np.pi*b.frequency
    story=floor*omega**2*2/(3-np.sqrt(5))*np.array([[2,-1],[-1,1]])
    for j,extra in ((0,added),(2,added),(4,0),(6,added)):
        M[j:j+2,j:j+2]=np.diag([floor,floor+extra])
        K[j:j+2,j:j+2]=story
        C[j:j+2,j:j+2]=2*b.damping/omega*story
    a=np.radians(d.angle+180*d.reverse)
    rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    cross=rot@coupling
    M[[1,3],8:]=cross;M[8:,[1,3]]=cross.T
    M[8:,8:]=fluid_mass;K[8:,8:]=fluid_k
    excitation=np.zeros(size);excitation[[0,1,4,5,6,7]]=1
    eff=np.linalg.inv(M); load=M@excitation
    # 전체 손실계수를 바닥 관 길이에 비례해 분배한다. 형상별 실측 보정값은 아니다.
    weights=.5*1000*d.area*d.loss*np.linalg.norm(ends-starts,axis=1)/d.B
    def resistance(v):
        flow=T@v[8:]
        return T.T@(weights*abs(flow)*flow)
    q=np.zeros(size);v=np.zeros(size)
    Q=np.full((len(t),size),np.nan);V=Q.copy();Q[0]=q;V[0]=v
    fmax=np.sqrt(np.linalg.eigvals(eff@K).real.max())
    stepmax=min(.01,2*np.pi/(100*fmax))
    nvalid=1;failure=None;last_h=None
    for i,span in enumerate(np.diff(t),1):
        count=max(1,int(np.ceil(span/stepmax-1e-10)));dt=span/count;h=dt/2
        if last_h is None or abs(last_h-h)>1e-13:
            inv=np.linalg.inv(M+h*C+h*h*K)
            vm=inv@M;qm=-h*inv@K;force=-h*inv@load;drag=h*inv[:,8:]
            # 1~3개의 가지 유속만을 미지수로 두는 중점법의 비선형 저항 방정식
            transfer=T@drag[8:]@T.T
            last_h=h
        for j in range(count):
            gm=ag[i-1]+(ag[i]-ag[i-1])*(j+.5)/count
            free=vm@v+qm@q+force*gm
            rhs=T@free[8:];flow=rhs.copy()
            for _ in range(20):
                residual=flow+transfer@(weights*abs(flow)*flow)-rhs
                if max(abs(residual))<1e-11*(1+max(abs(rhs))):break
                flow-=np.linalg.solve(np.eye(len(flow))+transfer*(2*weights*abs(flow)),residual)
            else:raise ValueError('유체 저항 계산이 수렴하지 않았습니다. 입력 조건을 확인하세요.')
            mid=free-drag@(T.T@(weights*abs(flow)*flow))
            q=q+dt*mid;v=2*mid-v
            rise=S@q[8:]
            if min(rise)<=-leg or max(rise)>=d.freeboard/up[0,2]:
                failure=t[i-1]+(j+1)*dt;break
        if failure is not None:break
        Q[i]=q;V[i]=v;nvalid+=1
    force=-Q@K.T-V@C.T
    currents=V[:,8:]@T.T
    force[:,8:]-=(weights*abs(currents)*currents)@T
    acc=force@eff.T
    heights=(Q[:,8:]@S.T)*up[:,2]
    speeds=(V[:,8:]@S.T)*up[:,2]
    dt=np.diff(t);vg=np.r_[0,np.cumsum(dt*(ag[:-1]+ag[1:])/2)]
    ground=np.r_[0,np.cumsum(dt*vg[:-1]+dt**2*(2*ag[:-1]+ag[1:])/6)]
    travel=max(np.linalg.norm(Q[:nvalid,[0,2]],axis=1).max(),np.linalg.norm(Q[:nvalid,[1,3]],axis=1).max())+max(abs(ground[:nvalid]))
    return dict(t=t,ag=ag,ground=ground,first=Q[:,[0,2]],travel=travel,installed=True,
        levels=heights,rest_height=leg*up[0,2],
        state=np.column_stack([Q[:,1],V[:,1],heights[:,-1],speeds[:,-1],Q[:,5],V[:,5],Q[:,7],V[:,7]]),
        acc=acc[:,[5,7,1]],lateral=np.column_stack([Q[:,3],V[:,3]]),lateral_acc=acc[:,3],
        valid=failure is None,failure_time=failure,n_valid=nvalid)


def simulate_space(b, d, t, ag, installed=True):
    """수평 두 방향의 지붕 운동과 두 수면의 연직 운동을 결합한다.

    건물의 강성과 감쇠가 두 수평축에서 같다고 가정한다. 관과 나란한 축에서는
    물과 건물이 힘을 주고받고, 직각 축에서는 물도 건물에 실린 질량으로 움직인다.
    축별로 적분한 뒤 지진 방향 X와 횡방향 Y로 다시 회전한다.
    """
    validate(b, d)
    t, ag = np.asarray(t, dtype=float), np.asarray(ag, dtype=float)
    if (t.ndim != 1 or t.shape != ag.shape or len(t) < 2
            or not np.all(np.isfinite(t)) or not np.all(np.isfinite(ag))
            or np.any(np.diff(t) <= 0)):
        raise ValueError('시간은 증가해야 하며 각 시간에 가속도 하나가 필요합니다.')
    if installed and d.shape!='U형': return network_response(b,d,t,ag)
    theta = math.radians(d.angle)
    co, si = math.cos(theta), math.sin(theta)
    # 동일한 두 층 질량·층강성. 사용자가 설정한 값은 빈 건물의 1차 고유진동수다.
    floor_mass=b.mass/2
    omega=2*math.pi*b.frequency
    story_k=floor_mass*omega**2*2/(3-math.sqrt(5))
    story=np.array([[2.,-1.],[-1.,1.]])*story_k
    # 좌표 0,1,2: 관 방향의 1층·지붕·물. 3,4: 고정질량 건물. 5,6: 빈 건물.
    added_mass = d.mass+b.container_mass if installed else 0.
    M=np.diag([floor_mass,floor_mass+added_mass,d.mass,
               floor_mass,floor_mass+added_mass,floor_mass,floor_mass])
    # 관 제거 시 부착 질량과 물의 반력을 동시에 제거한다.
    M[1,2]=M[2,1]=1000*d.area*d.B if installed else 0.
    K=np.zeros((7,7)); C=np.zeros_like(K)
    for j in (0,3,5):
        K[j:j+2,j:j+2]=story
        C[j:j+2,j:j+2]=story*(2*b.damping/omega)
    K[2,2]=2*1000*d.area*G
    drag=.5*1000*d.area*d.loss
    base=np.array([co,co,0.,1.,1.,1.,1.])
    load=M@base
    freq=math.sqrt(np.max(np.linalg.eigvals(np.linalg.solve(M,K)).real))
    step_limit=min(.01,2*math.pi/(100*freq))
    q=np.zeros(7);v=np.zeros(7)
    Q=np.full((len(t),7),np.nan); V=Q.copy()
    Q[0]=q; V[0]=v
    previous=None;failure=None;n_valid=1
    for i,span in enumerate(np.diff(t),1):
        count=max(1,math.ceil(span/step_limit-1e-10))
        dt=span/count;h=dt/2
        if previous is None or abs(previous-h)>1e-13:
            inv=np.linalg.inv(M+h*C+h*h*K)
            vcoef=inv@M; qcoef=inv@(-h*K); forcing=inv@(-h*load)
            loss=inv[:,2]*h*drag
            previous=h
        for j in range(count):
            gm=ag[i-1]+(ag[i]-ag[i-1])*(j+.5)/count
            free=vcoef@v+qcoef@q+forcing*gm
            w=2*free[2]/(1+math.sqrt(1+4*loss[2]*abs(free[2])))
            mid=free-loss*(abs(w)*w)
            q=q+dt*mid;v=2*mid-v
            if abs(q[2])>=d.limit:
                failure=float(t[i-1]+(j+1)*dt);break
        if failure is not None: break
        Q[i]=q;V[i]=v;n_valid+=1
    def world(values,j):
        # 관에 직각인 축은 선형 고정 질량 건물의 응답을 투영한다.
        return co*values[:,j]+si*si*values[:,j+3], si*(values[:,j]-co*values[:,j+3])
    x,y=world(Q,1); vx,vy=world(V,1)
    force=-Q@K.T-V@C.T
    force[:,2]-=drag*np.abs(V[:,2])*V[:,2]
    absolute=np.linalg.solve(M,force.T).T
    ax,ay=world(absolute,1)
    # 정지 출발. 샘플 사이에 선형인 입력 가속도를 구간별로 정확히 두 번 적분한다.
    dt=np.diff(t)
    vg=np.r_[0,np.cumsum(dt*(ag[:-1]+ag[1:])/2)]
    ground=np.r_[0,np.cumsum(dt*vg[:-1]+dt**2*(2*ag[:-1]+ag[1:])/6)]
    first=np.column_stack(world(Q,0))
    roof=np.column_stack((x,y))
    travel=max(np.nanmax(np.linalg.norm(roof[:n_valid],axis=1)),
               np.nanmax(np.linalg.norm(first[:n_valid],axis=1)))+np.max(np.abs(ground[:n_valid]))
    return dict(t=t,ag=ag,ground=ground,first=first,travel=travel,installed=installed,
                state=np.column_stack((x,vx,Q[:,2],V[:,2],Q[:,6],V[:,6],Q[:,4],V[:,4])),
                acc=np.column_stack((absolute[:,6],absolute[:,4],ax)),
                lateral=np.column_stack((y,vy)),lateral_acc=ay,valid=failure is None,
                failure_time=failure,n_valid=n_valid)


def export_space(folder, b, d, r, title):
    plt = setup_plot()
    if r.get('installed',True):
        fig, axes = plt.subplots(3,2,figsize=(14,10))
        fig.subplots_adjust(hspace=.65,wspace=.25)
        draw_result(fig, list(axes.flat), b,d,r,title)
    else:
        fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
        for series,label in [(r['ground'],'지면'),(r['ground']+r['first'][:,0],'1층'),(r['ground']+r['state'][:,0],'지붕')]:
            ax.plot(r['t'],series*1000,label=label)
        ax.set(title='관 없음 · '+title,xlabel='시간 (초)',ylabel='정지 기준 위치 (mm)')
        ax.legend();ax.grid(axis='y',alpha=.2)
    save_result(folder,b,d,r,fig,title)
    plt.close(fig)
    if 'levels' in r:
        np.savetxt(Path(folder)/'column_heights.csv',np.column_stack((r['t'],r['rest_height']+r['levels'])),
            delimiter=',',comments='',header='time_s,'+','.join(f'column_{j+1}_height_m' for j in range(r['levels'].shape[1])))
    np.savetxt(Path(folder)/'spatial_motion.csv',
               np.column_stack((r['t'],r['ground'],r['first'],r['state'][:,0],r['lateral'][:,0],
                                (r['rest_height']+r['levels'][:,0] if 'levels' in r else d.h-r['state'][:,2]) if r.get('installed',True) else np.full(len(r['t']),np.nan),
                                (r['rest_height']+r['levels'][:,-1] if 'levels' in r else d.h+r['state'][:,2]) if r.get('installed',True) else np.full(len(r['t']),np.nan),
                                r['acc'][:,2],r['lateral_acc'])),
               header='time_s,ground_displacement_m,floor1_relative_X_m,floor1_relative_Y_m,roof_X_m,roof_Y_m,left_water_height_m,right_water_height_m,absolute_ax_m_s2,absolute_ay_m_s2',
               delimiter=',',comments='')


@dataclass(frozen=True)
class Source:
    east: float = -6
    north: float = 0
    depth: float = 8
    strength: float = .03

    def input_for(self, waveform, heading):
        """거리·방향 영향을 독립적으로 실험하는 가상 입력. 시간은 도착 후 기준."""
        if not all(math.isfinite(x) for x in asdict(self).values()) or self.depth <= 0 or self.strength < 0:
            raise ValueError('진원 깊이는 양수, 기준 세기는 0 이상이어야 합니다.')
        distance = math.hypot(self.east,self.north,self.depth)
        bearing = math.degrees(math.atan2(-self.north,-self.east)) if self.east or self.north else 0.
        # 10 km에서 입력한 진폭. 근거리 발산을 막도록 1 km 완화 길이를 둔다.
        pga = self.strength*math.sqrt(101)/math.hypot(distance,1)
        peak = np.max(np.abs(waveform))
        if peak == 0 and pga > 0:
            raise ValueError('모든 값이 0인 기록은 세기를 키울 수 없습니다.')
        signal = waveform*(pga*G/peak) if peak else np.zeros_like(waveform)
        return signal, (heading-bearing)%180, dict(**asdict(self),distance_km=distance,
            bearing_deg=bearing,heading_deg=heading,site_pga_g=pga)


def lab(b,d,t,ag,title,out):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import time
    from concurrent.futures import ThreadPoolExecutor
    root = tk.Tk()
    root.title('TLCD 실험실')
    root.geometry('1070x745')
    root.minsize(830,620)
    panel = ttk.Frame(root,padding=14)
    panel.pack(side='left',fill='y')
    body = ttk.Frame(root,padding=(0,10,12,8))
    body.pack(side='left',fill='both',expand=True)
    ttk.Label(panel,text='TLCD 실험실',font=('Malgun Gothic',17,'bold')).pack(anchor='w')
    settings=ttk.Notebook(panel);settings.pack(fill='x',pady=6)
    pipe_page=ttk.Frame(settings,padding=6); quake_page=ttk.Frame(settings,padding=6)
    shape_page=ttk.Frame(settings,padding=6)
    settings.add(shape_page,text='모양')
    shape_var=tk.StringVar(value=d.shape)
    ttk.Label(shape_page,text='관 모양').pack(anchor='w')
    picker_shape=ttk.Combobox(shape_page,textvariable=shape_var,values=SHAPES,state='readonly',width=19)
    picker_shape.pack(fill='x')
    tilt_var=tk.StringVar(value=str(d.tilt))
    tr=ttk.Frame(shape_page);tr.pack(fill='x',pady=4)
    ttk.Label(tr,text='V형 경사각 (°)').pack(side='left')
    ttk.Entry(tr,textvariable=tilt_var,width=7).pack(side='right')
    ttk.Label(shape_page,text='같은 물을 각 기둥에 나눠 채웁니다.\n모양 비교 시 빈 관 질량도 고정합니다.').pack(anchor='w')
    settings.add(pipe_page,text='관 설정');settings.add(quake_page,text='진원 · 세기')
    freq_page=ttk.Frame(settings,padding=6)
    settings.add(freq_page,text='각진동수')
    omega_b=tk.StringVar(value=f'{2*math.pi*b.frequency:.9g}')
    omega_l=tk.StringVar(value=f'{2*math.pi*d.frequency:.9g}')
    tune_mode=tk.BooleanVar(value=False)
    for label,var in [('건물 ω (rad/s)',omega_b),('TLCD ω (rad/s)',omega_l)]:
        row=ttk.Frame(freq_page);row.pack(fill='x',pady=4)
        ttk.Label(row,text=label).pack(side='left')
        entry=ttk.Entry(row,textvariable=var,width=11);entry.pack(side='right')
    def switch_tune():
        entry.configure(state='normal' if tune_mode.get() else 'readonly')
        if not tune_mode.get(): preview_frequency()
    ttk.Checkbutton(freq_page,text='TLCD 목표값으로 물 높이 조절',variable=tune_mode,command=switch_tune).pack(anchor='w')
    entry.configure(state='readonly')
    ttk.Label(freq_page,text='건물은 질량을 유지하고 강성을 조절합니다.\nTLCD는 바닥 관 길이를 유지합니다.').pack(anchor='w')
    original_wave=ag
    reference_peak=float(np.max(np.abs(ag))/G)
    src=Source(strength=reference_peak)
    quake_fields={}
    for key,label in [('east','동쪽 위치 (km)'),('north','북쪽 위치 (km)'),('depth','깊이 (km)'),('strength','10 km 기준 세기 (g)')]:
        row=ttk.Frame(quake_page);row.pack(fill='x',pady=2)
        ttk.Label(row,text=label).pack(side='left')
        quake_fields[key]=tk.StringVar(value=str(round(getattr(src,key),12)))
        ttk.Entry(row,textvariable=quake_fields[key],width=10).pack(side='right')
    ttk.Label(quake_page,text='건물 위치 (0, 0) · 음수는 서쪽/남쪽\n세기 = 기준 거리의 최대 지면 가속도').pack(anchor='w')
    pipe_on=tk.BooleanVar(root,True)
    fields={}
    for key,label,value in [('B','바닥 관 총길이 (cm)',d.B*100),('h','기준 U형 물 높이 (cm)',d.h*100),
              ('diameter','관 안지름 (cm)',d.diameter*100),('angle','설치 각도 (°)',d.angle),
              ('loss','관의 손실계수',d.loss)]:
        row=ttk.Frame(pipe_page); row.pack(fill='x',pady=3)
        ttk.Label(row,text=label).pack(side='left')
        fields[key]=tk.StringVar(value=f'{value:g}')
        ttk.Entry(row,textvariable=fields[key],width=8).pack(side='right')
    ttk.Label(pipe_page,text='관 방향은 동쪽 0° · 북쪽 90°').pack(anchor='w',pady=4)
    def preview_frequency(*_):
        if tune_mode.get(): return
        try:
            length=(float(fields['B'].get())+2*float(fields['h'].get()))/100
            trial=replace(d,shape=shape_var.get(),B=float(fields['B'].get())/100,h=float(fields['h'].get())/100,tilt=float(tilt_var.get()))
            omega_l.set(f'{2*np.pi*trial.frequency:.9g}' if min(trial.B,trial.h,trial.tilt)>0 else '')
        except (ValueError,ZeroDivisionError): omega_l.set('')
    for key in ('B','h'): fields[key].trace_add('write',preview_frequency)
    def change_shape(_):
        # 모양 전환은 마지막으로 적용한 물의 양을 기준으로 한다.
        previous=current[0]
        for key in ('B','h','diameter'):
            fields[key].set(str(100*getattr(previous,key)))
        tune_mode.set(False);switch_tune()
        calculate(keep_time=True)
    picker_shape.bind('<<ComboboxSelected>>',change_shape)
    tilt_var.trace_add('write',preview_frequency)
    status=tk.StringVar()
    summary=tk.StringVar()
    source=tk.StringVar(value=title)
    ttk.Label(body,textvariable=source).pack(anchor='w')
    mode_bar=ttk.Frame(body)
    mode_bar.pack(fill='x',pady=(3,4))
    ttk.Label(mode_bar,text='지붕 장치',font=('Malgun Gothic',10,'bold')).pack(side='left',padx=(0,8))
    mode_buttons=[]
    for label,value in [('관 있음',True),('관 없음',False)]:
        btn=ttk.Radiobutton(mode_bar,text=label,value=value,variable=pipe_on,command=lambda:calculate(keep_time=True))
        btn.pack(side='left',padx=5);mode_buttons.append(btn)
    def save_view():
        from matplotlib.figure import Figure
        from matplotlib.patches import Polygon,Ellipse,Rectangle
        path=filedialog.asksaveasfilename(title='현재 시점을 발표용 그림으로 저장',
            defaultextension='.png',filetypes=[('PNG 그림','*.png')],initialfile='TLCD_실험장면.png')
        if not path: return
        setup_plot()
        width,height=scene.winfo_width(),scene.winfo_height()
        fig=Figure(figsize=(width/90,height/90),facecolor=scene['bg'])
        ax=fig.add_axes((0,0,1,1),xlim=(0,width),ylim=(height,0));ax.axis('off')
        for layer,item in enumerate(scene.find_all()):
            kind=scene.type(item)
            pts=np.array(scene.coords(item)).reshape(-1,2)
            opt=lambda key:scene.itemcget(item,key)
            color=opt('fill') or 'none'
            if kind=='text':
                anchor=opt('anchor')
                ax.text(*pts[0],opt('text'),color=color,fontsize=8,zorder=layer,
                        ha='left' if 'w' in anchor else 'center',va='top' if 'n' in anchor else 'bottom' if 's' in anchor else 'center')
            elif kind=='line':
                ax.plot(*pts.T,color=color,lw=float(opt('width'))*.8,zorder=layer,ls='--' if opt('dash') else '-')
                if opt('arrow')=='last' and np.linalg.norm(pts[-1]-pts[-2])>1:
                    ax.annotate('',xy=pts[-1],xytext=pts[-2],zorder=layer,
                                arrowprops=dict(arrowstyle='-|>',color=color,lw=1.8,shrinkA=0,shrinkB=0))
            else:
                style=dict(facecolor=color,edgecolor=opt('outline') or 'none',lw=.7,zorder=layer)
                if kind=='polygon': patch=Polygon(pts,**style)
                elif kind=='rectangle': patch=Rectangle(pts[0],*(pts[1]-pts[0]),**style)
                elif kind=='oval': patch=Ellipse(pts.mean(axis=0),*(pts[1]-pts[0]),**style)
                else: continue
                ax.add_patch(patch)
        fig.savefig(path,dpi=210)
        status.set('현재 장면을 PNG로 저장했어요.')
    ttk.Button(mode_bar,text='장면 PNG 저장',command=save_view).pack(side='right')
    scene=tk.Canvas(body,bg='#f1f5f7',height=335,highlightthickness=1,highlightbackground='#d5dfe5')
    scene.pack(fill='both',expand=True,pady=(4,0))
    bar=ttk.Frame(body); bar.pack(fill='x',pady=5)
    play=ttk.Button(bar); play.pack(side='left')
    position=tk.DoubleVar(value=0)
    scrub=ttk.Scale(bar,from_=0,to=1,variable=position)
    scrub.pack(side='left',fill='x',expand=True,padx=8)
    clock=tk.StringVar(); ttk.Label(bar,textvariable=clock,width=10).pack(side='left')
    graph_type=tk.StringVar(value='건물 변위')
    choices=['건물 변위','건물 가속도','물 높이 변화','옆 방향 변위','지면 · 1층 · 지붕 위치']
    chooser=ttk.Combobox(body,state='readonly',values=choices,textvariable=graph_type,width=18)
    chooser.pack(anchor='w')
    chart=tk.Canvas(body,height=142,bg='white',highlightthickness=0); chart.pack(fill='x')
    legend=tk.StringVar()
    ttk.Label(body,textvariable=legend).pack(anchor='w')
    pool=ThreadPoolExecutor(max_workers=1)
    current=[d,simulate_space(b,d,t,ag)]
    current[1]['source']=src.input_for(ag,d.angle)[2]
    history=[]
    camera=[math.radians(36),math.radians(25)]
    mesh_cache={}
    playback={'running':False,'start':0.,'busy':False}
    cache={(src,d,b,True):current[1]}
    # 재생 중 같은 도형은 좌표만 바꾼다. 카메라를 돌려도 매 프레임 새로 만들지 않는다.
    objects={}
    def paint(index,kind,coords,**style):
        key=(index,kind)
        item=objects.get(key)
        if item is None:
            item=getattr(scene,'create_'+kind)(*coords,**style)
            objects[key]=item
        else:
            scene.coords(item,*coords)
            if kind=='text': scene.itemconfigure(item,text=style['text'])
    def draw_network(dam,r,j,P,shift,roof,heading,yaw,pitch):
        # 세계 방위에 놓은 원형 관의 겉면과 실제 계산된 물 높이를 투영한다.
        key=('network',dam,heading,tuple(P.ravel()))
        if key not in mesh_cache:
            mesh_cache.clear()
            for obj in list(objects):
                if isinstance(obj[0],tuple) and obj[0][0]=='tube':
                    scene.delete(objects.pop(obj))
            feet,up,leg,starts,ends=network(dam)[:5]
            az=np.radians(heading)
            rot=np.array([[np.cos(az),np.sin(az),0],[-np.sin(az),np.cos(az),0],[0,0,1]])
            tubes=[]
            segments=[(a,b-a,None) for a,b in zip(starts,ends)]
            segments.extend((a,u*(leg+dam.freeboard/u[2]),i) for i,(a,u) in enumerate(zip(feet,up)))
            direction=np.array([np.sin(yaw)*np.cos(pitch),np.cos(yaw)*np.cos(pitch),np.sin(pitch)])
            for a,v,column in segments:
                start=a@rot+[0,0,roof+dam.diameter/2]
                vector=v@rot;axis=vector/np.linalg.norm(vector)
                normal=np.cross(axis,[0,0,1] if abs(axis[2])<.98 else [0,1,0])
                normal/=np.linalg.norm(normal);normal2=np.cross(axis,normal)
                phi=np.arange(16)*2*np.pi/16
                ring=dam.diameter/2*(np.cos(phi)[:,None]*normal+np.sin(phi)[:,None]*normal2)
                tubes.append(((start+.5*vector)@direction,start@P,vector@P,ring@P,column))
            mesh_cache[key]=(sorted(tubes,key=lambda v:v[0]),leg,up[0,2])
        tubes,leg,slope=mesh_cache[key]
        for index,(_,base,span,ring,column) in enumerate(tubes):
            base=base+shift
            fill=1 if column is None else (leg*slope+r['levels'][j,column])/(leg*slope+dam.freeboard)
            low=base+.86*ring;high=low+span*fill
            # 실루엣의 볼록 다각형은 두 원을 이은 원통의 정확한 선형 투영이다.
            length=np.linalg.norm(span)
            perp=np.array([-span[1],span[0]])/max(length,1e-12)
            k=int(np.argmax(ring@perp));k2=(k+8)%16
            def line_id(tag,coords,**style):
                paint(('tube',index,tag),'line',coords.ravel().tolist(),**style)
            body=np.array([low[k],high[k],high[k2],low[k2]])
            paint(('tube',index,'body'),'polygon',body.ravel().tolist(),fill='#298bb4',outline='')
            for tag,center in [('start',low),('end',high)]:
                paint(('tube',index,tag),'polygon',center.ravel().tolist(),fill='#55b8ce',outline='')
            for side in [k,k2]:
                line_id(('side',side==k),np.vstack([base+ring[side],base+span+ring[side]]),fill='#758e9e',width=1)
            if column is not None:
                rim=base+span+ring
                line_id('rim',np.vstack([rim,rim[0]]),fill='#486678',width=2)
                paint(('tube',index,'number'),'text',tuple(base+span+[0,-9]),text=str(column+1),fill='#284353')

    def draw_scene(*_):
        dam,r=current
        last=r['n_valid']-1
        j=min(last,int(np.searchsorted(t,position.get())))
        x=float(r['state'][j,0]); y=float(r['lateral'][j,0]); level=r['state'][j,2]
        width,height=scene.winfo_width(),scene.winfo_height()
        if width<5 or height<5: return
        # 세계 좌표는 동·북·위. 건물과 카메라는 진원 방위와 독립적이다.
        meta=r['source']
        bearing=math.radians(meta['bearing_deg'])
        dx=x*math.cos(bearing)-y*math.sin(bearing)
        dy=x*math.sin(bearing)+y*math.cos(bearing)
        bx,by,H=.28,.22,.42
        yaw,pitch=camera
        cy,sy,cp,sp=math.cos(yaw),math.sin(yaw),math.cos(pitch),math.sin(pitch)
        ground=r['ground'][j]
        gx,gy=ground*math.cos(bearing),ground*math.sin(bearing)
        fx,fy=r['first'][j]
        first_x=fx*math.cos(bearing)-fy*math.sin(bearing)
        first_y=fx*math.sin(bearing)+fy*math.cos(bearing)
        # 바닥부터 관 끝까지의 범위와 이번 실험의 전체 이동량으로 한 번의 배율을 정한다.
        half=max(.54,dam.B*.5+dam.diameter)+r['travel']
        top=H+dam.h+dam.freeboard+dam.diameter
        scale=min(width*.65/(2.05*half),(height-62)/(top*cp+2*half*sp+.06))
        P=np.array([[cy,sy*sp],[-sy,cy*sp],[0,-cp]])*max(1,scale)
        fixed=np.array([width*.35,26+(top*cp+half*sp)*scale])
        origin=fixed+np.array([gx,gy,0])@P
        def poly(key,points,fill,outline):
            screen=np.asarray(points)@P+origin
            paint(key,'polygon',screen.ravel().tolist(),fill=fill,outline=outline)
        # 배경 기준선은 정지하고, 흙과 콘크리트 기초는 적분한 지면 변위만큼 함께 이동한다.
        ref=np.array([[-.5,-.32,-.03],[.5,-.32,-.03]])@P+fixed
        paint('stationary','line',ref.ravel().tolist(),fill='#8799a3',width=2,dash=(3,4))
        def slab(name,hx,hy,z,color):
            poly(name,[(-hx,-hy,z),(hx,-hy,z),(hx,hy,z),(-hx,hy,z)],color,'#899292')
        slab('soil',.42,.32,-.026,'#c3bfaf')
        slab('footing',bx+.025,by+.025,-.006,'#a6b2b5')
        for n in range(5):
            mark=np.array([[-.37+n*.18,.28,-.025],[-.35+n*.18,.28,-.025]])@P+origin
            paint(('soilmark',n),'line',mark.ravel().tolist(),fill='#897657',width=2)
        def wallpoint(a,f,z):
            part=z/H*2
            if part<=1: ux,uy=part*first_x,part*first_y
            else: ux,uy=first_x+(part-1)*(dx-first_x),first_y+(part-1)*(dy-first_y)
            return (a+ux,f+uy,z)
        def face(key,side,u1,u2,z1,z2,color,edge):
            def point(u,z):
                return wallpoint(u,math.copysign(by,cy),z) if side=='front' else wallpoint(math.copysign(bx,sy),u,z)
            poly(key,[point(u1,z1),point(u2,z1),point(u2,z2),point(u1,z2)],color,edge)
        for side,half,base in [('front',bx,'#dbe4e8'),('side',by,'#aabac4')]:
            for story in range(2):
                face((side,'wall',story),side,-half,half,H*story/2,H*(story+1)/2,base,'#8496a0')
                for n in range(3):
                    u=(-.67+n*.67)*half
                    face((side,story,n),side,u-.18*half,u+.18*half,
                         (.15+.50*story)*H,(.40+.50*story)*H,'#497488','#e9eff0')
            face(('band',side),side,-half,half,.48*H,.52*H,'#788f9d','#788f9d')
        face('door','front',-.15*bx,.15*bx,0,.29*H,'#3c5663','#d3cbbd')
        poly('roof',[wallpoint(a,f,H) for a,f in [(-bx,-by),(bx,-by),(bx,by),(-bx,by)]],
             '#d4dce0','#72838b')
        # 둥근 단면을 따라 만든 표면 격자. 정지 형상과 카메라 투영은 재사용한다.
        if r.get('installed',True) and dam.shape!='U형':
            draw_network(dam,r,j,P,origin+np.array([dx,dy,0])@P,H,meta['heading_deg'],yaw,pitch)
        elif r.get('installed',True):
            angle=math.radians(meta['bearing_deg']+dam.angle)
            key=(dam,round(angle,9),width,height,yaw,pitch,scale)
            if key not in mesh_cache:
                mesh_cache.clear()
                radius=dam.diameter/2
                bend=min(dam.B*.22,dam.h*.45,radius*1.8)
                # (관 축을 따른 수평 위치, 높이, 단면의 평면 내 법선)
                rings=[(-dam.B/2,dam.h+dam.freeboard,1.,0.)]
                for alpha in np.linspace(0,math.pi/2,5):
                    rings.append((-dam.B/2+bend*(1-math.cos(alpha)),bend*(1-math.sin(alpha)),math.cos(alpha),math.sin(alpha)))
                for alpha in np.linspace(0,math.pi/2,5):
                    rings.append((dam.B/2-bend+bend*math.sin(alpha),bend*(1-math.cos(alpha)),-math.sin(alpha),math.cos(alpha)))
                rings.append((dam.B/2,dam.h+dam.freeboard,-1.,0.))
                ca,sa=math.cos(angle),math.sin(angle)
                theta=np.arange(16)*2*math.pi/16
                def make_mesh(rad):
                    out=[]
                    for u,z,nu,nz in rings:
                        along=u+rad*np.cos(theta)*nu
                        across=rad*np.sin(theta)
                        out.append(np.column_stack([along*ca-across*sa,along*sa+across*ca,
                                H+radius+z+rad*np.cos(theta)*nz]))
                    return np.array(out)
                water=make_mesh(radius*.87); shell=make_mesh(radius)
                # 면 그리기 순서는 고정 카메라에서 먼 쪽부터.
                depth=shell@np.array([sy*cp,cy*cp,sp])
                faces=sorted(((i,k) for i in range(len(rings)-1) for k in range(16)),
                             key=lambda pair:sum(depth[ii,kk%16] for ii in (pair[0],pair[0]+1) for kk in (pair[1],pair[1]+1)))
                ix=np.array([[i,i+1,i+1,i] for i,k in faces])
                kx=np.array([[k,k,(k+1)%16,(k+1)%16] for i,k in faces])
                shades=['#%02x%02x%02x'%(18+int(v*22),115+int(v*70),159+int(v*62)) for v in (.5+.5*np.sin(theta))]
                mesh_cache[key]=(water@P,shell@P,ix,kx,[shades[k] for i,k in faces])
                # 회전한 관의 면 순서와 음영이 바뀌므로 관 그림만 새로 만든다.
                for obj in list(objects):
                    if isinstance(obj[0],tuple) and obj[0][0]=='tube':
                        scene.delete(objects.pop(obj))
            water,shell,ix,kx,colors=mesh_cache[key]
            shift=origin+np.array([dx,dy,0])@P
            water=water+shift
            water[0]+=( -dam.freeboard-level)*P[2]
            water[-1]+=(-dam.freeboard+level)*P[2]
            for n,coords in enumerate(water[ix,kx].reshape(-1,8).tolist()):
                paint(('tube','water',n),'polygon',coords,fill=colors[n],outline='')
            shell=shell+shift
            for side in (0,-1):
                paint(('tube','surface',side),'polygon',water[side].ravel().tolist(),fill='#81deeb',outline='#418ca3')
                ring=shell[side]
                paint(('tube','rim',side),'line',np.vstack([ring,ring[0]]).ravel().tolist(),fill='#566f81',width=2)
            for k in (0,4,8,12):
                paint(('tube','wall',k),'line',shell[:,k].ravel().tolist(),fill='#839ba8',width=1)
        elif mesh_cache:
            mesh_cache.clear()
            for obj in list(objects):
                if isinstance(obj[0],tuple) and obj[0][0]=='tube':
                    scene.delete(objects.pop(obj))
        # 진원은 km 단위이므로 건물과 별도 축척의 평면도·단면도에 표시한다.
        mx,my=width*.83,height*.33
        size=min(width*.105,height*.14)
        east,north,depth=meta['east'],meta['north'],meta['depth']
        span=max(1.,abs(east),abs(north))
        factor=size/span
        sx,sy=mx+factor*east,my-factor*north
        paint('mapcard','rectangle',(mx-size-22,my-size-26,mx+size+22,my+size+23),fill='#ffffff',outline='#d4dfe5')
        paint('mapx','line',(mx-size,my,mx+size,my),fill='#d5dfe5')
        paint('mapy','line',(mx,my-size,mx,my+size),fill='#d5dfe5')
        # 지도 화살표의 시작은 진원 바로 위의 점, 끝은 건물이다.
        paint('map-arrow','line',(sx,sy,mx,my),fill='#d4473e',width=3,arrow='last',arrowshape=(9,11,4))
        paint('site','rectangle',(mx-4,my-4,mx+4,my+4),fill='#18394b',outline='')
        paint('surface-point','oval',(sx-4,sy-4,sx+4,sy+4),fill='#d4473e',outline='white')
        name='건물 바로 아래' if not (east or north) else ('북' if north>0 else '남' if north<0 else '')+('동' if east>0 else '서' if east<0 else '')+'쪽'
        paint('map-caption','text',(mx,my+size+12),text=f'● 진앙   ■ 건물',fill='#344f60')
        # 건물 모서리 아래에 가려지지 않도록 지붕 위에 독립적인 방향 지시선을 둔다.
        uv=(np.array([math.cos(bearing),math.sin(bearing),0.])@P)
        uv=uv/max(1e-12,np.linalg.norm(uv))
        target=np.array([width*.33,42.])
        start=target-uv*min(width*.09,46.)
        overhead=not(east or north)
        if overhead: start=target-np.array([0.,20.])
        paint('direction','line',(*start,*target),fill='#d4473e',width=3,arrow='last',arrowshape=(9,11,4))
        paint('arrival','text',(width*.51,40),text=f'{name}에서 도달' if not overhead else '건물 바로 아래 진원',fill='#943e39')
        paint('map-title','text',(mx,my-size-13),text='진앙 → 건물  |  위쪽은 북쪽',fill='#304c5c')
        surface=height*.65
        paint('ground','line',(mx-size,surface,mx+size,surface),fill='#69858e')
        section_scale=size/max(1.,abs(east),depth)
        sx=mx+east*section_scale
        sz=surface+section_scale*depth
        paint('depthline','line',(sx,surface,sx,sz),fill='#cf8985',dash=(3,3))
        paint('hypocenter','oval',(sx-5,sz-5,sx+5,sz+5),fill='#d43d39',outline='')
        paint('section','text',(mx,surface-12),text='동서 단면 · 빨간 점은 진원',fill='#304c5c')
        paint('depth','text',(mx,sz+15),text=f"깊이 {meta['depth']:g} km",fill='#913a34')
        paint('header','text',(12,10),text=('관 있음' if r.get('installed',True) else '관 없음')+'   /   드래그로 시점 회전',anchor='nw',fill='#233d50')
        paint('readout','text',(12,height-8),text=f'{t[j]:.2f} 초   |   지면 {ground*1000:+.1f} mm   ·   지면 대비 지붕 {x*1000:+.1f} mm',anchor='sw',fill='#314b5a')
        clock.set(f'{t[j]:.2f} 초')
        chart.delete('cursor')
        cw=chart.winfo_width()
        px=65+(cw-85)*(t[j]-t[0])/(t[-1]-t[0])
        chart.create_line(px,14,px,117,fill='#8392a0',dash=(3,3),tags='cursor')

    def draw_chart(*_):
        dam,r=current
        idx=choices.index(graph_type.get())
        off=not r.get('installed',True)
        if off and idx in (0,1): arrays=r['state'][:,0:1]*1000 if idx==0 else r['acc'][:,2:3]/G
        elif idx==0: arrays=r['state'][:,[4,6,0]]*1000
        elif idx==1: arrays=r['acc']/G
        elif idx==2: arrays=r.get('levels',r['state'][:,2:3])*1000
        elif idx==3: arrays=r['lateral'][:,:1]*1000
        else:
            ground=r['ground']
            arrays=1000*np.column_stack((ground,ground+r['first'][:,0],ground+r['state'][:,0]))
        if idx==4: legend.set('정지 기준 위치   ·   회색  지면       주황  1층       파랑  지붕')
        elif idx in (0,1): legend.set('회색  장치 없음       주황  같은 무게 고정       파랑  TLCD')
        else: legend.set('파랑  TLCD를 설치한 모형')
        if off and idx!=4: legend.set('관 없음 · 건물 응답' if idx!=2 else '관 없음 · 물 높이 측정 대상 없음')
        unit=['mm','g','mm','mm','mm'][idx]
        n=r['n_valid']; width=chart.winfo_width()
        chart.delete('all')
        if width<100: return
        stride=max(1,n//max(80,width//2))
        ids=np.unique(np.r_[np.arange(0,n,stride),n-1])
        bound=max(float(np.max(np.abs(arrays[:n]))),1e-10)
        xx=65+(width-85)*(t[ids]-t[0])/(t[-1]-t[0])
        colors=['#798898','#d18534','#007db4'] if arrays.shape[1]==3 else ['#007db4']
        for sign in [-1,-.5,0,.5,1]:
            pos=65-sign*48
            chart.create_line(65,pos,width-20,pos,fill='#e4eaf0')
            chart.create_text(61,pos,text=f'{bound*sign:.2g} {unit}',anchor='e')
        if idx==2 and 'levels' in r:
            colors=['#256ca2','#dd882f','#24836b','#99568a'][:arrays.shape[1]]
            legend.set('기둥 1 파랑 · 2 주황 · 3 초록 · 4 보라' if len(colors)==4 else ' · '.join(['기둥 1 파랑','2 주황','3 초록'][:len(colors)]))
        for i,color in enumerate(colors):
            if off and idx==2: break
            yy=65-48*arrays[ids,i]/bound
            if len(xx)>1: chart.create_line(*np.column_stack([xx,yy]).ravel().tolist(),fill=color,width=1.6)
        chart.create_text(65,130,text='0 초',anchor='w')
        chart.create_text(width-20,130,text=f'{t[-1]:g} 초',anchor='e')
        draw_scene()
    def refresh():
        dam,r=current
        m=metrics(r)
        def pct(v): return '기준값 0' if v is None else f'{v:+.1f}%'
        if not r.get('installed',True):
            summary.set(f"관 없음 · 기준 실험\n최대 지붕 변위  {1000*np.max(np.abs(r['state'][:,0])):.2f} mm\n최대 절대가속도  {np.max(np.abs(r['acc'][:,2]))/G:.3f} g")
        elif m:
            a,_,z=m['peak_displacement_m']
            summary.set(f'최대 변위 · 지진 방향\n{a*1000:.2f} → {z*1000:.2f} mm\n\n장치 없음 대비 저감률\n변위  {pct(m["peak_reduction_vs_bare_pct"])}\n가속도  {pct(m["acceleration_reduction_vs_bare_pct"])}\n\n{dam.shape} · 물 {dam.mass:.3f} kg\n관 포함 {dam.mass+b.container_mass:.3f} kg')
        else:
            summary.set(f'{r["failure_time"]:.2f}초에 수면 한계 도달\n여기까지 재생할 수 있어요.')
        scrub.configure(to=max(t[r['n_valid']-1],1e-12))
        position.set(0); draw_chart()
    def pause():
        playback['running']=False
        play.configure(text='▶ 재생')
    def tick():
        if not playback['running']: return
        end=t[current[1]['n_valid']-1]
        pos=min(end,time.monotonic()-playback['start'])
        position.set(pos); draw_scene()
        if pos>=end: pause()
        else: playback['timer']=root.after(33,tick)
    def toggle():
        if playback['running']:
            pause()
            if playback.get('timer'): root.after_cancel(playback['timer'])
            return
        end=t[current[1]['n_valid']-1]
        if position.get()>=end: position.set(0)
        playback.update(running=True,start=time.monotonic()-position.get())
        play.configure(text='Ⅱ 멈춤'); tick()
    def task(work,done):
        if playback['busy']: return
        pause()
        if playback.get('timer'): root.after_cancel(playback['timer'])
        playback['busy']=True
        picker_shape.state(['disabled'])
        for button in mode_buttons: button.state(['disabled'])
        status.set('계산 중…')
        job=pool.submit(work)
        def poll():
            if not job.done():
                root.after(40,poll); return
            playback['busy']=False
            picker_shape.state(['!disabled'])
            for button in mode_buttons: button.state(['!disabled'])
            try: done(job.result())
            except Exception as error:
                status.set('조건을 확인해 주세요.'); messagebox.showerror('실험실',str(error))
        root.after(40,poll)
    def calculate(keep_time=False):
        nonlocal ag
        previous_time=position.get() if keep_time else 0.
        enabled=pipe_on.get()
        try:
            vals={key:float(field.get()) for key,field in fields.items()}
            for key in ['B','h','diameter']: vals[key]/=100
            wb=float(omega_b.get())
            if not math.isfinite(wb) or wb<=0: raise ValueError('건물 각진동수는 양수여야 합니다.')
            target_b=replace(b,frequency=wb/(2*math.pi))
            vals.update(shape=shape_var.get(),tilt=float(tilt_var.get()))
            if tune_mode.get():
                wl=float(omega_l.get())
                if not math.isfinite(wl) or wl<=0: raise ValueError('TLCD 각진동수는 양수여야 합니다.')
                lo,hi=1e-7,max(1.,vals['h'])
                def rate(height):
                    trial=replace(d,**dict(vals,h=height))
                    validate(target_b,trial)
                    return 2*np.pi*trial.frequency
                if rate(lo)<=wl: raise ValueError('이 관 길이에서 가능한 각진동수보다 큽니다.')
                while rate(hi)>wl:
                    hi*=2
                    if hi>1e6:raise ValueError('모형 크기에 맞는 각진동수를 입력하세요.')
                for _ in range(45):
                    mid=(lo+hi)/2
                    if rate(mid)>wl:lo=mid
                    else:hi=mid
                vals['h']=(lo+hi)/2
            raw=replace(d,**vals); validate(target_b,raw)
            quake=Source(**{key:float(field.get()) for key,field in quake_fields.items()})
            signal,relative,meta=quake.input_for(original_wave,raw.angle)
            dam=replace(raw,angle=relative,reverse=(raw.angle-meta['bearing_deg'])%360>=180)
            key=(quake,raw,target_b,enabled)
        except ValueError as error:
            pipe_on.set(current[1].get('installed',True))
            messagebox.showerror('입력 확인',str(error)); return
        def done(r):
            if len(cache)>=6: cache.clear()
            nonlocal ag,b
            ag=signal
            b=target_b
            fields['h'].set(f'{dam.h*100:.9g}')
            cache[key]=r; r['source']=meta; current[:]=[dam,r]
            source.set(f"진원 거리 {meta['distance_km']:.1f} km   ·   건물 입력 {meta['site_pga_g']:.4f} g   ·   관과 진동 방향 {relative:.1f}°")
            refresh()
            position.set(min(previous_time,t[r['n_valid']-1]));draw_scene()
            status.set('계산 완료 · 비교 목록에 담을 수 있어요.')
        task(lambda:cache[key] if key in cache else simulate_space(target_b,dam,t,signal,enabled),done)
    def save():
        path=filedialog.askdirectory(title='저장할 폴더 선택')
        if path:
            from datetime import datetime
            folder=Path(path)/datetime.now().strftime('TLCD_%Y%m%d_%H%M%S_%f')
            dam,r=current
            run_title=source.get()
            task(lambda:export_space(folder,b,dam,r,run_title),lambda _:status.set(f'저장 완료\n{folder}'))
    def compare():
        from datetime import datetime
        folder=out/datetime.now().strftime('compare_%Y%m%d_%H%M%S_%f')
        dam=current[0]
        def show(_):
            status.set(f'28개 조건 저장 완료\n{folder}')
            from PIL import Image,ImageTk
            win=tk.Toplevel(root);win.title('형상과 설치 각도 비교')
            with Image.open(folder/'sweep.png') as img:
                img.thumbnail((980,470)); photo=ImageTk.PhotoImage(img)
            label=ttk.Label(win,image=photo);label.image=photo;label.pack()
        task(lambda:sweep(b,dam,t,ag,folder),show)
    def angle_window():
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure
        # 창을 연 시점의 실험 조건을 보관한다. 비교 중에는 설치 각도만 바뀐다.
        dam = current[0]
        wave = ag
        fixed_building = b
        setup_plot()
        win = tk.Toplevel(root)
        win.title('설치 각도별 지붕 변위')
        win.geometry('920x670')
        ttk.Label(win, text=f'{source.get()}   |   바닥 관 {dam.B*100:g} cm   ·   물 높이 {dam.h*100:g} cm').pack(anchor='w', padx=14, pady=8)
        controls = ttk.Frame(win)
        controls.pack(fill='x', padx=14)
        ttk.Label(controls, text='진동 방향과 관 사이 각도 (°)').pack(side='left')
        angles_text = tk.StringVar(win, value='0, 45, 90')
        entry = ttk.Entry(controls, textvariable=angles_text, width=25)
        entry.pack(side='left', padx=7)
        feedback = tk.StringVar(win)
        fig = Figure(figsize=(8, 3.8), constrained_layout=True)
        axis = fig.subplots()
        plot = FigureCanvasTkAgg(fig, master=win)
        table = ttk.Treeview(win, columns=('angle','peak','rate'), show='headings', height=5)
        for key, label in [('angle','설치 각도'),('peak','최대 변위 (mm)'),('rate','장치 없음 대비 저감률')]:
            table.heading(key, text=label)
            table.column(key, width=145, anchor='center')
        memo = {}
        def redraw():
            try:
                angles = sorted(set(float(v.strip()) for v in angles_text.get().split(',')))
                if not 2 <= len(angles) <= 8 or not all(0 <= a <= 180 for a in angles):
                    raise ValueError
            except ValueError:
                feedback.set('0~180 사이의 서로 다른 각도 2~8개를 쉼표로 나눠 입력하세요.')
                return
            if playback['busy']:
                feedback.set('이전 계산이 끝나면 다시 비교를 누르세요.')
                return
            feedback.set('계산 중…')
            def compute():
                # 큰 기록을 여러 번 비교해도 메모리에는 현재 선택만 보관한다.
                return {a: memo[a] if a in memo else simulate_space(fixed_building, replace(dam, angle=a), t, wave) for a in angles}
            def display(results):
                status.set('각도별 비교 완료')
                if not win.winfo_exists(): return
                memo.clear()
                memo.update(results)
                table.delete(*table.get_children())
                axis.clear()
                reference = max(results.values(), key=lambda r: r['n_valid'])
                axis.plot(t, reference['state'][:,4]*1000, '--', color='#56616b', label='장치 없음', lw=1.1)
                for angle, result in results.items():
                    axis.plot(t, result['state'][:,0]*1000, label=f'{angle:g}°', lw=1.5)
                    m = metrics(result)
                    if m is None:
                        values = ('—', f"{result['failure_time']:.2f}초 수면 한계")
                    else:
                        rate = m['peak_reduction_vs_bare_pct']
                        values = (f"{1000*m['peak_displacement_m'][2]:.2f}", '기준값 0' if rate is None else f'{rate:+.1f}%')
                    table.insert('', 'end', values=(f'{angle:g}°', *values))
                table.configure(height=len(angles))
                axis.set(xlabel='시간 (초)', ylabel='지진 방향 지붕 변위 (mm)', xlim=(t[0],t[-1]))
                axis.grid(alpha=.2)
                axis.legend(ncol=min(5,len(angles)+1))
                plot.draw()
                feedback.set('각 실선은 해당 각도의 TLCD입니다. 양의 저감률은 흔들림이 줄었다는 뜻이에요.')
            task(compute,display)
        ttk.Button(controls, text='그래프 비교', command=redraw).pack(side='left')
        entry.bind('<Return>', lambda _: redraw())
        ttk.Label(win, textvariable=feedback).pack(anchor='w', padx=14, pady=5)
        table.pack(side='bottom', fill='x', padx=14, pady=8)
        toolbar = NavigationToolbar2Tk(plot, win, pack_toolbar=False)
        toolbar.pack(side='bottom', fill='x')
        plot.get_tk_widget().pack(fill='both', expand=True)
        redraw()

    def remember():
        dam,r=current
        if any(previous is r for _,previous in history):
            status.set('이미 담긴 실험이에요. 조건을 바꾸고 적용하세요.');return
        m=r['source']
        name=(f"실험 {len(history)+1}  |  ω 건물 {2*math.pi*b.frequency:.2f} / 물 {2*math.pi*dam.frequency:.2f} rad/s  |  진원 ({m['east']:g}, {m['north']:g}, 깊이 {m['depth']:g}) km"
              f"  ·  기준 {m['strength']:g} g  ·  관 {m['heading_deg']:g}°"
              f"  ·  길이/물높이/지름 {dam.B*100:g}/{dam.h*100:g}/{dam.diameter*100:g} cm  ·  손실 {dam.loss:g}")
        mode=dam.shape if r.get('installed',True) else '관 없음'
        if mode=='관 없음': name=f"실험 {len(history)+1} | 진원 ({m['east']:g}, {m['north']:g}, 깊이 {m['depth']:g}) km · 기준 {m['strength']:g} g · 건물 {2*math.pi*b.frequency:.2f} rad/s"
        history.append((mode+' · '+name,r))
        status.set(f'실험 {len(history)} 저장됨 · 실험 그래프 비교에서 선택하세요.')

    def history_window():
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
        from matplotlib.figure import Figure
        if not history: remember()
        setup_plot()
        win=tk.Toplevel(root);win.title('실험 그래프 비교');win.geometry('1040x700')
        ttk.Label(win,text='함께 볼 실험을 클릭하세요. 다시 클릭하면 선택이 풀려요.').pack(anchor='w',padx=12,pady=5)
        items=tk.Listbox(win,selectmode='multiple',exportselection=False,height=5)
        items.pack(fill='x',padx=12)
        rows=list(history)
        for label,_ in rows: items.insert('end',label)
        items.select_set(max(0,len(rows)-3),'end')
        bar=ttk.Frame(win);bar.pack(fill='x',padx=12,pady=5)
        options=['건물 변위','건물 가속도','물 높이 변화','옆 방향 변위','지면 입력 가속도','장치 없는 건물 변위','고정 질량 건물 변위']
        var=tk.StringVar(value=options[0])
        picker=ttk.Combobox(bar,textvariable=var,values=options,state='readonly',width=24)
        picker.pack(side='left')
        fig=Figure(figsize=(9,4.3),constrained_layout=True)
        ax=fig.subplots();canvas=FigureCanvasTkAgg(fig,master=win)
        toolbar=NavigationToolbar2Tk(canvas,win,pack_toolbar=False);toolbar.pack(side='bottom',fill='x')
        note=tk.StringVar();ttk.Label(win,textvariable=note,wraplength=900).pack(side='bottom',fill='x',padx=12)
        canvas.get_tk_widget().pack(fill='both',expand=True)
        def redraw(*_):
            ax.clear();ax.set_facecolor('#f5f8fa')
            kind=options.index(var.get());unit='g' if kind in (1,4) else 'mm'
            peaks=[]
            for j in items.curselection():
                _,r=rows[j]
                if kind==4: yy=r['ag']/G
                elif kind==1: yy=r['acc'][:,2]/G
                elif kind==3: yy=r['lateral'][:,0]*1000
                else: yy=r['state'][:,{0:0,2:2,5:4,6:6}[kind]]*1000
                if kind==2 and not r.get('installed',True):
                    peaks.append(f'실험 {j+1} · 관 없음');continue
                ax.plot(t,yy,label=f"실험 {j+1} · {'관 있음' if r.get('installed',True) else '관 없음'}",lw=1.7)
                if r['valid'] or kind==4: peaks.append(f'실험 {j+1}   {np.nanmax(np.abs(yy)):.3g} {unit}')
                else: peaks.append(f"실험 {j+1}   {r['failure_time']:.2f}초 수면 한계")
            ax.set(xlabel='진동 도착 후 시간 (초)',ylabel=f'{var.get()} ({unit})',xlim=(t[0],t[-1]))
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#d6e2e7')
            if peaks: ax.legend(ncol=min(5,len(peaks)))
            note.set('최대 절댓값     '+'     |     '.join(peaks))
            canvas.draw_idle()
        def sync():
            for label,r in history[len(rows):]:
                rows.append((label,r));items.insert('end',label);items.select_set('end')
            redraw()
        ttk.Button(bar,text='새로 담은 실험 가져오기',command=sync).pack(side='left',padx=7)
        items.bind('<<ListboxSelect>>',redraw);picker.bind('<<ComboboxSelected>>',redraw);redraw()

    def model_info():
        messagebox.showinfo('실험 모델',
            '모양 선택만 바꾸면 물 총부피와 빈 관 질량이 유지됩니다. 바닥 총길이 B를 가지에 나누고, 기준 U형 물 높이 h의 두 배를 모든 다리에 나눕니다. V형은 수평 기준 경사각이며 피자 거치대는 120도 간격 3개, 다기둥형은 일렬 4개입니다. 모든 관은 같은 안지름입니다. 그림의 차단판 효과는 손실계수로 근사합니다.\n\n'
            '진원 좌표는 건물을 기준으로 입력합니다. 가상 지진은 진원에서 건물로 향하는 수평선 방향으로 흔들린다고 가정합니다. 바로 아래 진원은 동쪽을 기준으로 사용합니다.\n\n'
            '세기는 진원 거리 10 km에서의 최대 가속도(g)입니다. 같은 파형에 거리 계수 √101 / √(거리²+1)을 곱합니다(거리 단위 km). 모든 기록은 진동 도착 시점부터 비교합니다. 실제 지반, 지진 규모, 파의 도착시간은 계산 범위 밖입니다.\n\n'
            '물은 관을 따라 움직이는 한 물기둥으로 계산합니다. 물의 관성, 두 수면 높이 차에 따른 중력, 유속의 제곱에 비례하는 저항이 들어갑니다. 3D 화면은 이 계산값을 보여줍니다.')
    for label,cmd in [('조건 적용',calculate),('현재 실험 담기',remember),('실험 그래프 비교',history_window),('각도별 그래프 비교',angle_window),('형상 × 각도 비교',compare),('결과 저장',save),('물과 지진 계산 원리',model_info)]:
        ttk.Button(panel,text=label,command=cmd).pack(fill='x',pady=3)
    ttk.Separator(panel).pack(fill='x',pady=12)
    ttk.Label(panel,textvariable=summary,justify='left').pack(anchor='w')
    ttk.Label(panel,textvariable=status,wraplength=205,foreground='#215a72').pack(anchor='w',pady=12)
    play.configure(command=toggle)
    scrub.configure(command=lambda _:draw_scene())
    def stop_scrub(_):
        pause()
        if playback.get('timer'): root.after_cancel(playback['timer'])
    scrub.bind('<ButtonPress-1>',stop_scrub,add='+')
    drag_start=[0,0]
    def orbit(event):
        camera[0]-=(event.x-drag_start[0])*.011
        camera[1]=min(1.20,max(.14,camera[1]+(event.y-drag_start[1])*.009))
        drag_start[:]=[event.x,event.y]
        draw_scene()
    scene.bind('<ButtonPress-1>',lambda e:drag_start.__setitem__(slice(None),[e.x,e.y]))
    scene.bind('<B1-Motion>',orbit)
    scene.bind('<Configure>',draw_scene);chart.bind('<Configure>',draw_chart)
    chooser.bind('<<ComboboxSelected>>',draw_chart)
    pause();refresh();status.set('기본 실험 준비 완료 · ▶ 재생')
    root.protocol('WM_DELETE_WINDOW',root.destroy)
    try: root.mainloop()
    finally: pool.shutdown(wait=False,cancel_futures=True)


def main():
    parser=argparse.ArgumentParser(description='TLCD 3D 실험실')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--sweep',action='store_true')
    parser.add_argument('--input',type=Path)
    parser.add_argument('--unit',choices=['g','m/s2','cm/s2'])
    parser.add_argument('--config',type=Path)
    # 파일로 실행하면 파일 옆에, 코드를 대화형 창에 붙여 넣으면 현재 폴더에 저장한다.
    script_path = globals().get('__file__')
    from_file = bool(script_path and not str(script_path).startswith('<'))
    base_folder = Path(script_path).resolve().parent if from_file else Path.cwd()
    parser.add_argument('--out',type=Path,default=base_folder/'results_3d')
    args=parser.parse_args(None if from_file else [])
    # 결과 파일만 만드는 Matplotlib은 비대화형으로 둔다. 애니메이션은 Tk가 그린다.
    import matplotlib
    matplotlib.use('Agg')
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else {}
    b,d=Building(**cfg.get('building',{})),Damper(**cfg.get('damper',{}))
    validate(b,d)
    if args.input and not args.unit: parser.error('CSV에는 --unit 단위가 필요합니다.')
    t,ag=load_input(args.input,args.unit) if args.input else recorded_input()
    title=args.input.name if args.input else '1940 엘센트로 남북 기록  ·  31.16초  ·  모형 입력 0.03 g'
    if args.sweep: sweep(b,d,t,ag,args.out)
    if args.headless: export_space(args.out,b,d,simulate_space(b,d,t,ag),title)
    else: lab(b,d,t,ag,title,args.out)


if __name__=='__main__':
    main()


