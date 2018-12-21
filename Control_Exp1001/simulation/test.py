import numpy as np

def f1(ki,qi,ci,ut,A3,cu,qu,p,h3,theta,rho):
    x1 = ki*qi*ci*ut*A3
    x2 = x1 + (ki*qi*ci-cu)*qu
    x3 = x2*(ki*qi*ci+cu)
    x4 = p*h3*A3*(ki*qi*ci+cu)-ci*qi*theta*rho

    y1=ci*qi*theta*rho*(ki*qi*ci*ut*A3+(ki*qi*ci-cu)*qu)
    y2 = -A3*p*(ki*qi*ci+cu)*(p*h3*A3*(ki*qi*ci+cu)-ci*qi*theta*rho)
    return x3/x4,y1/y2

def f2(ki,qi,ci,ut,A3,cu,qu,p,h3,theta,rho):
    wt=ci*qi
    cl=ki*qi*ci
    ca=p*(cl+cu)
    R2=ca
    R3=h3
    ur=qu/A3
    R4=cl*(ur+ut) - cu*ur
    R1=-wt*theta/A3*rho/(ca*ca)
    X=R1*R4/(R1*R2+R3)
    Y=R4/(R1*R2+R3)
    dcu = Y/p
    return (dcu,X)

if __name__ == '__main__':
    for _ in range(10):
        a = [np.random.random() for _ in range(11)]
        print(f1(*a))
        print(f2(*a))
        print()


