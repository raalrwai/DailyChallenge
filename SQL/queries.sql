SELECT e.department, e.name, e.salary
FROM Employee e
JOIN (
    SELECT department, MAX(salary) AS max_salary
    FROM Employee
    GROUP BY department
) m ON e.department = m.department AND e.salary = m.max_salary;

