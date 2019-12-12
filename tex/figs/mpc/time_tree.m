clear;clc;
t=0:10

figure(1);clf;
hold on
stem(2^0*t,1+0*t,'k.','markersize',20)
stem(2^0*t,0*t,'k.','markersize',20)
stem(2^-1*t,2+0*t,'k.','markersize',20)
stem(2^-1*t,0*t,'k.','markersize',20)
stem(2^-2*t,3+0*t,'k.','markersize',20)
stem(2^-2*t,0*t,'k.','markersize',20)
axis off
set(gcf,'color','w')
pbaspect([10 1 1])
saveas(gcf,'time_tree.png')

t=0:100
figure(2);clf;
stem(exp(t*0.25),0*t,'k.','markersize',20)
axis off
set(gcf,'color','w')
pbaspect([10 1 1])
saveas(gcf,'exp_sample.png')