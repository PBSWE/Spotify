-- 0. Drop table INSTRUCTOR in case it already exists
drop table INSTRUCTOR
;
--1. Create table INSTRUCTOR
CREATE TABLE INSTRUCTOR
  (ins_id INTEGER PRIMARY KEY NOT NULL, 
   lastname VARCHAR(15) NOT NULL, 
   firstname VARCHAR(15) NOT NULL, 
   city VARCHAR(15), 
   country CHAR(2)
  )
;
--2A. Insert single row for Phil Balta
INSERT INTO INSTRUCTOR
  (ins_id, lastname, firstname, city, country)
  VALUES 
  (1, 'Balta', 'Phil', 'Vegas', 'US')
;
--2B. Insert the two rows for Joe and Mary
INSERT INTO INSTRUCTOR
  VALUES
  (2, 'Smith', 'Joe', 'Paris', 'FR'),
  (3, 'Jane', 'Mary', 'London', 'UK')
;
--3. Select all rows in the table
SELECT * FROM INSTRUCTOR
;
--4. Change the city for Phil to Boston
UPDATE INSTRUCTOR SET city='Boston' where ins_id=1
;
--5. Delete the row for Mary Jane
DELETE FROM INSTRUCTOR where ins_id=3
;;
