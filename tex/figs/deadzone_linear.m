
clear;clc;
x=-2:0.01:2
a=1
y = hinge(x-a)-hinge(-x-a)

figure('Renderer', 'painters', 'Position', [10 10 300 100])
plot(x,y,'k','linewidth',3)
text(a,y(1),'b')
text(-a,y(1),'a')
% text(x(1),0,'0')
% text(0,y(1),'0')
hold on
plot(x,0*x,'k')
plot(0*x,x/2,'k')
text(x(end),0,'x')
% text(0,y(end),'  (x-a)_+-(-x-a)_+')
axis off
set(gcf,'color','w')
saveas(gcf,'deadzone_linear.png')

function y = hinge(x)
    y = 0 * x;
    for i=1:length(x)
        if x(i) > 0
            y(i) = x(i);
        end
    end
end