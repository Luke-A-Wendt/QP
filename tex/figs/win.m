clear;clc;
x=-2:0.01:2
a=-1
b=1
y = window(x,a,b)

figure('Renderer', 'painters', 'Position', [10 10 300 100])
plot(x,y,'k','linewidth',3)
text(a,-0.3,'a')
text(b,-0.3,'b')
hold on
plot(x,0*x,'k')
plot(0*x,x/2,'k')
text(x(end),0,'x')
% text(0,y(end)+.5,'  sat(x,a,b)')
axis off
set(gcf,'color','w')
saveas(gcf,'win.png')

function y = window(x,a,b)
    for i=1:length(x)
        if x(i) < a
            y(i) = 0;
        elseif x(i) > b
            y(i) = 0;
        else
            y(i) = 1
        end
    end
end