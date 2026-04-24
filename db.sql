drop database if exists music_rec;
create database music_rec;
use music_rec;

create table users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    email VARCHAR(50), 
    password VARCHAR(50),
    username VARCHAR(20),
    age INT,
    gender VARCHAR(10)
    );
