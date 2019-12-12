
clear;clc;
x=-2:0.01:2
a=-1
b=1
y = saturate(x,a,b)

figure('Renderer', 'painters', 'Position', [10 10 300 100])
plot(x,y,'k','linewidth',3)
text(-0.2,a,'a')
text(-0.2,b,'b')
hold on
plot(x,0*x,'k')
plot(0*x,x/2,'k')
text(x(end),0,'x')
% text(0,y(end)+.5,'  sat(x,a,b)')
axis off
set(gcf,'color','w')
saveas(gcf,'sat.png')

function y = saturate(x,a,b)
    for i=1:length(x)
        if x(i) < a
            y(i) = a;
        elseif x(i) > b
            y(i) =b;
        else
            y(i) = x(i)
        end
    end
end