sizeo = .1;
a = 2*Pi;
Point(1) = {0, 0, 0, sizeo};
Point(2) = {a, 0, 0, sizeo};
Point(3) = {a, a, 0, sizeo};
Point(4) = {0, a, 0, sizeo};

Line(1) = {4, 3};
Line(2) = {3, 2};
Line(3) = {2, 1};
Line(4) = {1, 4};

Line Loop(9) = {4, 1, 2, 3};
Plane Surface(11) = {9};
Physical Surface(1) = {11};

Periodic Line {1} = {-3};
Periodic Line {2} = {-4};
